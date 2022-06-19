import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS, LOSSES, build_loss


def batch_argmax(tensor):
    '''
    Locate uv of max value in a tensor in dim 2 and dim 3.
    The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).

    Args:
        tensor: torch.tensor

    Returns: uv of max value with batch.

    '''
    # if tensor.shape = (16, 3, 64, 64)
    assert tensor.ndim == 4, 'The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).'
    b, c, h, w = tensor.shape
    tensor = tensor.reshape(b, c, -1)  # flatten the tensor to (16, 3, 4096)
    value, idx = tensor.max(dim=2)
    v = idx // w    # shape (16, 3)
    u = idx - v * w  # shape (16, 3)
    u, v = u.unsqueeze(-1), v.unsqueeze(-1)  # reshape u, v to (16, 3, 1) for cat
    uv = torch.cat((u, v), dim=-1)  # shape (16, 3, 2)
    return uv, value


def heatmap_to_uv(hm, mode='max'):
    '''
    Locate single keypoint pixel coordinate uv in heatmap.

    Args:
        hm:  shape (bz, c, w, h), dim=4
        mode: -

    Returns: keypoint pixel coordinate uv, shape  (1, 2)

    '''

    assert mode in ('max', 'average')
    if mode == 'max':
        uv = batch_argmax(hm)
    elif mode == 'average':
        b, c, h, w = hm.shape
        hm = hm.reshape(b, c, -1)
        hm = hm / torch.sum(hm, dim=-1, keepdim=True)
        v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
        u_map = u_map.reshape(1, 1, -1).float().to(hm.device)
        v_map = v_map.reshape(1, 1, -1).float().to(hm.device)
        u = torch.sum(u_map * hm, -1, keepdim=True)
        v = torch.sum(v_map * hm, -1, keepdim=True)
        uv = torch.cat((u, v), dim=-1)
    return uv


def batch_index_uv_value(tensor: torch.Tensor,
                   uv: torch.Tensor):
    '''
    example:
    N, C, H, W = 2, 3, 3, 3
    tensor = torch.randn(N, C, H, W)
    uv = torch.randint(H, (N, C, 2), dtype=torch.long)
    uv_value = batch_index_uv_value(tensor, uv)
    '''

    assert tensor.dim() == 4
    assert uv.dim() == 3 and uv.size(-1) == 2
    N, C, H, W = tensor.shape
    u =  uv[:, :, 0]
    v =  uv[:, :, 1]

    uv_flatten = v * H + u
    uv_flatten = uv_flatten.view(N, C, 1)
    tensor = tensor.view(N, C, H * W)
    uv_value =  torch.gather(tensor, -1 , uv_flatten)
    return uv_value




@LOSSES.register_module()
class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
    """

    def __init__(self, use_target_weight=False):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

@HEADS.register_module()
class GraspHead(nn.Module):
    def __init__(self,
                 loss_heatmap=dict(type='JointsMSELoss'),
                 loss_qual=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
                 loss_width=dict(type='MSELoss', loss_weight=1.0),
                 loss_quat=dict(type='MSELoss', loss_weight=1.0),
                 in_channels=256,
                 out_channels=10,
                 ):
        super().__init__()
        self.loss_qual = build_loss(loss_qual)
        self.loss_width = build_loss(loss_width)
        self.loss_quat = build_loss(loss_quat)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.hm_conv = nn.Sequential(*(ConvModule(in_channels, 512, 1),
                                   ConvModule(128, 64, 1),
                                   ConvModule(64, 32, 1),
                                   ConvModule(32, 16, 1),
                                   ConvModule(16, out_channels, 1)))

        self.grasp_conv = nn.Sequential(*(
                                   # ConvModule(3, 128, 1),
                                   ConvModule(in_channels, 128, 1),
                                   ConvModule(128, 64, 1),
                                   ConvModule(64, 32, 1),
                                   ConvModule(32, 16, 1),
                                   ConvModule(16, out_channels, 1)))


        self.conv_quat = nn.Sequential(
            ConvModule(10, 128, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(128, 256, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(256, 128, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(128, 10, 1, conv_cfg=dict(type='Conv1d')),
            nn.Linear(1, 4),
        )

        self.conv_qual = nn.Sequential(
            ConvModule(10, 128, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(128, 256, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(256, 128, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(128, 10, 1, conv_cfg=dict(type='Conv1d')),
            nn.Linear(1, 1),        )

        self.conv_width = nn.Sequential(
            ConvModule(10, 128, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(128, 256, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(256, 128, 1, conv_cfg=dict(type='Conv1d')),
            ConvModule(128, 10, 1, conv_cfg=dict(type='Conv1d')),
            nn.Linear(1, 1),        )

    def forward_train(self,
                      x,
                      # xyz,
                      gt_heatmap,
                      gt_qual,
                      gt_quat,
                      gt_width,
                      gt_valid_index=None,
                      **kwargs):
        pred_heatmap = self.hm_conv(x) # shape (bz, 10, 80, 80)
        pred_uv, _ = heatmap_to_uv(hm=pred_heatmap)
        # pred_uv = (pred_uv / 80 * 320).to(torch.int64)
        # grasp_feats = self.grasp_conv(xyz)
        grasp_feats = self.grasp_conv(x)

        pred_uv_value = batch_index_uv_value(grasp_feats, pred_uv)


        pred_qual = self.conv_qual(pred_uv_value)
        pred_quat = self.conv_quat(pred_uv_value)
        pred_width = self.conv_width(pred_uv_value)

        if gt_valid_index is not None:
            pred_qual, gt_qual = pred_qual[gt_valid_index], gt_qual[gt_valid_index]
            pred_quat, gt_quat = pred_quat[gt_valid_index], gt_quat[gt_valid_index]
            pred_width, gt_width = pred_width[gt_valid_index], gt_width[gt_valid_index]
            pred_heatmap, gt_heatmap = pred_heatmap[gt_valid_index], gt_heatmap[gt_valid_index]
        loss_qual = self.loss_qual(pred_qual, gt_qual)
        loss_quat = self.loss_quat(pred_quat, gt_quat)
        loss_width = self.loss_width(pred_width, gt_width)
        loss_heatmap = self.loss_heatmap(pred_heatmap, gt_heatmap)

        losses = dict(loss_width=loss_width,
                      loss_heatmap=loss_heatmap,
                      loss_quat=loss_quat,
                      loss_qual=loss_qual)
        return losses

    def simple_test(self, x):
        pred_heatmap = self.conv(x) # shape (bz, 10, 80, 80)
        pred_uv = heatmap_to_uv(hm=pred_heatmap)
        pred_uv_value = pred_heatmap[pred_uv] # shape (bz, 10, 1)

        pred_qual = self.fc_qual(pred_uv_value)
        pred_quat = self.fc_quat(pred_uv_value)
        pred_width = self.fc_width(pred_uv_value)

        res = dict(pred_uv=pred_uv,
                   pred_width=pred_width,
                   pred_qual=pred_qual,
                   pred_quat=pred_quat)

        return res


if __name__ == '__main__':
    x = torch.rand(16, 256, 80, 80)
    gt_heatmap =torch.rand(16, 10, 80, 80)
    gt_qual = torch.rand(16, 10)
    gt_quat = torch.rand(16, 10, 4)
    gt_width = torch.rand(16, 10)

    head = GraspHead()
    res = head.forward_train(x, gt_heatmap=gt_heatmap, gt_width=gt_width, gt_quat=gt_quat, gt_qual=gt_qual)

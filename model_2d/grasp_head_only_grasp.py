import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from mmdet.models.builder import HEADS
from model_2d.utils.utils import normalize_coordinate, ResnetBlockFC
from mmcv.cnn import ConvModule
from mmdet.models.builder import build_loss



class FeatsFusion(nn.Module):
    def __init__(self):
        super(FeatsFusion, self).__init__()
        cfg_2d = dict(act_cfg=dict(type='ReLU'),
                      norm_cfg=dict(type="BN", requires_grad=True),
                      conv_cfg=dict(type='Conv2d'))

        self.conv3 = nn.ConvTranspose2d(2048, 1024, 2)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 2)
        self.conv1 = nn.ConvTranspose2d(512, 256, 2)
        self.conv0 = nn.Sequential(
            ConvModule(256, 512, 1, **cfg_2d),
            ConvModule(512, 1024, 1, **cfg_2d),
            ConvModule(1024, 512, 1, **cfg_2d),
            ConvModule(512, 256, 1, **cfg_2d),
            ConvModule(256, 128, 1, **cfg_2d),
        )



    def forward(self,
                feats):
        feat0, feat1, feat2, feat3 = feats
        x3 = self.conv3(feat3)
        x2 = self.conv2(x3 + feat2)
        x1 = self.conv1(x2 + feat1)
        x0 = x1 + feat0

        out = self.conv0(x0)
        return out



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
    u =  uv[:, 0]
    v =  uv[:, 1]

    uv_flatten = v * H + u
    uv_flatten = uv_flatten.view(N, C, 1)
    tensor = tensor.view(N, C, H * W)
    uv_value =  torch.gather(tensor, -1 , uv_flatten)
    return uv_value


@HEADS.register_module()
class GraspHeadOnlyGrasp(nn.Module):
    def __init__(self,
                 # loss_heatmap=dict(type='JointsMSELoss'),
                 loss_qual=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
                 loss_width=dict(type='MSELoss', loss_weight=1.0),
                 loss_quat=dict(type='MSELoss', loss_weight=1.0),
                 in_channels=256,
                 out_channels=32,
                 ):
        super().__init__()
        self.loss_qual = build_loss(loss_qual)
        self.loss_width = build_loss(loss_width)
        self.loss_quat = build_loss(loss_quat)
        # self.loss_heatmap = build_loss(loss_heatmap)
        # self.hm_conv = nn.Sequential(*(ConvModule(in_channels, 512, 1),
        #                            ConvModule(128, 64, 1),
        #                            ConvModule(64, 32, 1),
        #                            ConvModule(32, 16, 1),
        #                            ConvModule(16, out_channels, 1)))

        cfg_2d = dict(
                    # act_cfg=dict(type='ReLU'),
                    #   norm_cfg=dict(type="BN"),
                      )
        cfg_1d = dict(
            # act_cfg=dict(type='ReLU'),
                      conv_cfg=dict(type='Conv1d'),
                      norm_cfg=None)

        self.grasp_conv = nn.Sequential(*(
                                   ConvModule(in_channels, 128, 1, **cfg_2d),
                                   ConvModule(128, 64, 1, **cfg_2d),
                                   ConvModule(64, 32, 1, **cfg_2d),
                                   ConvModule(32, 16, 1, **cfg_2d),
                                   ConvModule(16, out_channels, 1, **cfg_2d)))


        self.conv_quat = nn.Sequential(
            ConvModule(out_channels, 128, 1, **cfg_1d),
            ConvModule(128, 256, 1, **cfg_1d),
            ConvModule(256, 128, 1, **cfg_1d),
            ConvModule(128, 1, 1, **cfg_1d),
            nn.Linear(1, 4),
        )

        self.conv_qual = nn.Sequential(
            ConvModule(out_channels, 128, 1, **cfg_1d),
            ConvModule(128, 256, 1, **cfg_1d),
            ConvModule(256, 128, 1, **cfg_1d),
            ConvModule(128, 1, 1, **cfg_1d),
            nn.Linear(1, 1),        )

        self.conv_width = nn.Sequential(
            ConvModule(out_channels, 128, 1, **cfg_1d),
            ConvModule(128, 256, 1, **cfg_1d),
            ConvModule(256, 128, 1, **cfg_1d),
            ConvModule(128, 1, 1, **cfg_1d),
            nn.Linear(1, 1),        )

    def forward_train(self,
                      x,
                      pos,
                      gt_qual,
                      gt_quat,
                      gt_width):
        x = x[1]
        x_grasp = self.grasp_conv(x)
        # pos = (pos * 40).int() / 40

        # x_sampled = batch_index_uv_value(x_grasp, pos)
        #
        pos = pos * 2 - 1
        # F.grid_sample 中的第二个参数 需要范围[-1, 1]
        x_sampled = F.grid_sample(x_grasp, pos.reshape(pos.shape[0], 1, 1, -1), padding_mode='border', align_corners=True, mode='bilinear')

        # x_sampled = batch_index_uv_value(x_grasp, pos)
        x_sampled = x_sampled.squeeze(-1)
        pred_qual = self.conv_qual(x_sampled).flatten().sigmoid()
        pred_quat = self.conv_quat(x_sampled).squeeze(-2)
        pred_width = self.conv_width(x_sampled).flatten()

        loss_qual = self.loss_qual(pred_qual, gt_qual)
        # loss_qual = F.binary_cross_entropy(pred_qual, gt_qual)
        loss_quat = self.loss_quat(pred_quat, gt_quat)
        loss_width = self.loss_width(pred_width, gt_width)
        print(loss_qual)
        losses = dict(
            loss_width=loss_width,
                      loss_qual=loss_qual,
                      loss_quat=loss_quat)
        return losses

    def forward(self):
        pass




if __name__ == '__main__':
    x = torch.rand(32, 256, 80, 80)
    gt_grasp_map = torch.rand(32, 6, 80, 80)
    head = GraspHeadOnlyGrasp()
    res = head.forward_train(x, gt_grasp_map)
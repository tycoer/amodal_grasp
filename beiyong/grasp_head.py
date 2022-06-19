import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)


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

    def forward(self, output, target, target_weight):
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
                 in_channels=2048,
                 out_channels=20,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.loss_heatmap = JointsMSELoss(use_target_weight=True)


        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )
        self.grasp_conv = ConvModule(num_deconv_filters[-1], out_channels, 3, 2, 1)

        self.heatmap_conv = ConvModule(in_channels=num_deconv_filters[-1],
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=4,
                                       padding=0)
        self.fc_qual = nn.Linear(6400, 1)
        self.fc_width = nn.Linear(6400, 1)
        self.fc_quat = nn.Linear(6400, 4)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        pred_heatmap = self.heatmap_conv(x)
        x_grasp = self.grasp_conv(x)
        x_grasp = x_grasp.reshape(x_grasp.shape[0], x_grasp.shape[1], -1)
        pred_qual = self.fc_qual(x_grasp)
        pred_qual = torch.sigmoid(pred_qual).squeeze(-1)
        pred_quat = self.fc_quat(x_grasp)
        pred_width = self.fc_width(x_grasp).squeeze(-1)

        return pred_heatmap, pred_qual, pred_width, pred_quat

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding


    def _qual_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="mean")

    def _quat_loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduction="mean")

    def _width_loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduction="mean")


    def loss(self,
             pred_heatmap,
             pred_qual,
             pred_width,
             pred_quat,

             gt_heatmap,
             gt_qual,
             gt_width,
             gt_quat):

        loss_qual = self._qual_loss_fn(pred_qual, gt_qual)
        loss_heatmap = self.loss_heatmap(pred_heatmap, gt_heatmap, torch.ones((pred_heatmap.shape[0],
                                                                               pred_heatmap.shape[1],
                                                                               1,
                                                                               ), device=pred_heatmap.device))
        loss_width = self._width_loss_fn(pred_width, gt_width)
        loss_quat = self._quat_loss_fn(pred_quat, gt_quat)


        losses = dict(loss_heatmap= loss_heatmap,
                      loss_qual = loss_qual,
                      loss_width = loss_width,
                      loss_quat = loss_quat)
        return losses



if __name__ == '__main__':
    from mmdet.models.backbones.resnet import ResNet
    backbone = ResNet(depth=50)
    img = torch.rand(8, 3, 640, 640)
    feats = backbone(img)
    head = GraspHead(in_channels=2048,
                  out_channels=10,
                  )

    pred_heatmap, pred_qual, pred_width, pred_quat = head(feats[-1])

    gt_heatmap =  torch.rand_like(pred_heatmap)
    gt_qual = torch.rand_like(pred_qual)
    gt_quat = torch.rand_like(pred_quat)
    gt_width = torch.rand_like(pred_width)



    losses = head.loss(pred_heatmap,
                       pred_qual,
                       pred_width,
                       pred_quat,
                       gt_heatmap,
                       gt_qual,
                       gt_width,
                       gt_quat)
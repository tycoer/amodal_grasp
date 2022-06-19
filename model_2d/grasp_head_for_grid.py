import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from mmdet.models.builder import HEADS
from model_2d.utils.utils import normalize_coordinate, ResnetBlockFC
from mmcv.cnn import ConvModule
from mmdet.models.builder import build_loss

@HEADS.register_module()
class GripHead(nn.Module):
    def __init__(self,
                 loss_qual=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
                 loss_width=dict(type='MSELoss', loss_weight=1.0),
                 loss_quat=dict(type='MSELoss', loss_weight=1.0)):
        super().__init__()
        self.loss_qual = build_loss(loss_qual)
        self.loss_width = build_loss(loss_width)
        self.loss_quat = build_loss(loss_quat)

        self.conv = nn.Sequential(*(ConvModule(256, 128, 1),
                                    ConvModule(128, 64, 1),
                                    ConvModule(64, 32, 1),
                                    ConvModule(32, 6, 1)))

    def forward_train(self,
                      x,
                      gt_grasp_map):

        gt_qual_map = gt_grasp_map[:, 0]
        gt_width_map = gt_grasp_map[:, 1]
        gt_quat_map = gt_grasp_map[:, 2:]
        x = self.conv(x)
        pred_qual_map =  x[:, 0]
        pred_width_map = x[:, 1]
        pred_quat_map = x[:, 2:]

        loss_qual = F.binary_cross_entropy(pred_qual_map.sigmoid(), gt_qual_map)
        loss_width = self.loss_width(pred_width_map, gt_width_map)
        loss_quat = self.loss_quat(pred_quat_map, gt_quat_map)

        losses = dict(loss_qual=loss_qual,
                      loss_width=loss_width,
                      loss_quat=loss_quat)
        return losses

    def forward(self):
        pass


if __name__ == '__main__':
    x = torch.rand(32, 256, 80, 80)
    gt_grasp_map = torch.rand(32, 6, 80, 80)
    head = GripHead()
    res = head.forward_train(x, gt_grasp_map)
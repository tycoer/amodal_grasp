import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head.point_cloud_instance_seg_head_utils import *
from mmdet.models.builder import HEADS


class ins(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1_ins = nn.Linear(in_features=512, out_features=256)
    def forward_ins(self,
                    point_features,
                    global_features,
                    pred_bbox,
                    pred_bbox_score
                    ):
        bbox_num = pred_bbox.shape[1]
        p_num = 1


        global_features = torch.tile(F.relu(self.fc1_ins(global_features))[:,None,:], dims=[1, point_features.shape[1], 1, 1])
        point_features = F.relu(self.bn1(self.conv1(point_features[:,:,:,None])))
        bbox_info = torch.tile(torch.cat([pred_bbox.reshape(-1, bbox_num, 6),
                                          pred_bbox_score.permute(0, 2, 1)], axis=-1)[:,:,None,:],
                               [1, 1, point_features.shape[1], 1])
        ins_mask = torch.tile(point_features[:,None,:,:], [1, bbox_num, 1, 1])
        ins_mask = torch.cat([ins_mask, bbox_info])
        ins_mask = ins_mask.reshape(-1, p_num, int(ins_mask.shape[-1]), 1)

        pred_ins_mask = torch.sigmoid(ins_mask)
        return pred_ins_mask


def get_loss_bbscore(y_bbscore_pred, Y_bbvert):
    bb_num =Y_bbvert.shape[1]

    ##### helper -> the valid bbox
    Y_bbox_helper = torch.sum(Y_bbvert.view((-1, bb_num, 6)), dim=-1)
    Y_bbox_helper = torch.gt(Y_bbox_helper, 0.0).float()

    ##### bbox score loss
    bbox_loss_score = torch.mean(-Y_bbox_helper * torch.log(y_bbscore_pred + 1e-8)
                                     -(1. - Y_bbox_helper) * torch.log(1. - y_bbscore_pred + 1e-8))
    return bbox_loss_score


if __name__ == '__main__':
    # torch.manual_seed(0)
    # point_features = torch.rand(2, 64, 2048)
    # global_features = torch.rand(2, 1, 512)
    # pred_bbox = torch.rand(2, 10, 2, 3)
    # pred_bbox_score = torch.rand(2, 1, 10)
    pred_ins = torch.rand(2, 10, 2048)
    pc = torch.rand(2, 2048, 3)
    gt_ins = torch.rand(2, 10, 2048)



    loss = get_loss_pmask(pc, pred_ins, gt_ins)
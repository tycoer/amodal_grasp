import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head.point_cloud_instance_seg_head_utils import *
# from mmdet.models.builder import HEADS
#
# @HEADS.register_module()
class PointCloudInstanceSegHead(nn.Module):
    def __init__(self,
                 max_ins=10,
                 in_channels=512):
        super().__init__()
        # bbox
        self.fc1 = nn.Linear(in_features=in_channels, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=max_ins * 2 * 3)

        self.fc5 = nn.Linear(in_features=256, out_features=256)
        self.fc6 = nn.Linear(in_features=256, out_features=max_ins * 1)

        # sem_seg
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

        # ins_seg
        self.fc1_ins = nn.Linear(in_features=2048, out_features=64)
        self.conv1_ins = nn.Conv1d(4096, 2048, 1)
        self.fc2_ins = nn.Linear(in_features=2048, out_features=1)
        # self.conv2_ins = nn.Conv1d(4096, 2048, 1)
        # self.bn2_ins = nn.BatchNorm1d(128)
        #
        # self.conv3_ins = nn.Conv1d(256, 128, 1)
        # self.bn3_ins = nn.BatchNorm1d(128)

        # others
        self.max_ins = max_ins
        self.sem_loss = nn.BCELoss()


    def forward(self,
                point_features,
                global_features):
        # bbox
        b1 = F.relu(self.fc1(global_features))
        b2 = F.relu(self.fc2(b1))
        b3 = F.relu(self.fc3(b2))
        bbvert = self.fc4(b3)
        bbvert = bbvert.view((-1,self.max_ins ,2,3))
        points_min = torch.min(bbvert,dim=-2)[0][:, :, None, :]
        points_max = torch.max(bbvert,dim=-2)[0][:, :, None, :]
        pred_bbox = torch.cat([points_min,points_max],dim=-2)

        # bbox_score
        b4 = F.relu(self.fc5(b2))
        pred_bbox_score = torch.sigmoid(self.fc6(b4))

        # sem_seg
        x = self.drop1(F.relu(self.bn1(self.conv1(point_features))))
        x = self.conv2(x)
        pred_sem_seg = torch.sigmoid(x)


        # ins mask
        pred_ins_mask = self.forward_ins(point_features, global_features, pred_bbox, pred_bbox_score)
        return pred_bbox, pred_bbox_score, pred_sem_seg, pred_ins_mask


    def forward_ins(self,
                    point_features,
                    global_features,
                    pred_bbox,
                    pred_bbox_score
                    ):
        bbox_num = pred_bbox.shape[1]
        p_num = 2048
        p_f_num = 64
        bb_num = 10

        #global_features shape(bz, 1, 512)
        global_features = torch.tile(F.relu(self.fc1_ins(global_features)), dims=[1, p_num, 1])
        # global_features shape(bz, p_num, 64)
        point_features = torch.cat([point_features, global_features], dim=1)
        # point_features shape (bz, p_num, 64)
        point_features = F.relu(self.bn1(self.conv1_ins(point_features))) # shape  (bz, 256, 1)


        bbox_info = torch.tile(torch.cat([pred_bbox.reshape(-1, bbox_num, 6),  # shape (bz, max_ins, 6)
                                          pred_bbox_score.transpose(2, 1)], # shape (bz, max_ins, 1 )
                                          axis=-1)[:,:,None,:],
                               [1, 1, p_num, 1])
        # bbox_info shape(bz, max_ins, p_num, 7)
        # point_features shape (bz, 256, p_f_num)

        ins_mask = torch.tile(point_features[:,None,:,:], [1, bbox_num, 1, 1]) # shape (bz, max_ins, p_num, p_f_num)
        ins_mask = torch.cat([ins_mask, bbox_info], dim=-1)  # shape (bz, max_ins, p_num, p_f_num + 7)
        ins_mask = F.relu(self.bn2(self.conv2_ins(ins_mask)))

        ins_mask = F.relu(self.fc2_ins(ins_mask))  # shape (bz, max_ins, p_num, 1)
        ins_mask = ins_mask.reshape(-1, bbox_num, p_num)
        # ins_mask (bz, p_num, )
        pred_ins_mask = torch.sigmoid(ins_mask)
        return pred_ins_mask

    def forward_train(self,
                      # input
                      input_pc, # (bz, 2048, 3)
                      # backbone_features
                      global_features,
                      point_features,
                      # gt_info
                      gt_mask: torch.Tensor,
                      gt_bbox: torch.Tensor,
                      gt_ins_mask
                      ):
        input_pc = input_pc.permute(0, 2, 1)
        pred_bbox, pred_bbox_score, pred_sem_seg, pred_ins_mask = self.forward(point_features=point_features,
                                                                global_features=global_features)

        # pad bbox according to self.max_ins
        pad = torch.zeros((gt_bbox.shape[0],
                           self.max_ins - gt_bbox.shape[1],
                           2, 3), device=gt_bbox.device)
        gt_bbox = torch.cat((gt_bbox, pad), dim=1)

        # bbox part
        pred_bbox_cache, pred_bbox_border = bbvert_association(input_pc,
                                                               pred_bbox,
                                                               gt_bbox)
        loss_bbox = get_loss_bbvert(input_pc,
                                    pred_bbox_cache,
                                    gt_bbox)[0]
        # bbox_score part

        pred_bbox_score = bbscore_association(pred_bbox_score, pred_bbox_border)
        loss_bbox_score = get_loss_bbscore(pred_bbox_score,
                                           gt_bbox)
        # sem_seg part
        loss_sem_seg = self.sem_loss(pred_sem_seg, gt_mask)

        # ins part
        loss_ins_mask = get_loss_pmask(input_pc, pred_ins_mask, gt_ins_mask)


        # total loss
        losses = dict(loss_bbox=loss_bbox,
                      loss_bbox_score=loss_bbox_score,
                      loss_sem_seg=loss_sem_seg,
                      loss_ins_mask=loss_ins_mask)
        return losses

    def forward_test(self,
                     global_features,
                     point_features):
        return self.forward(global_features=global_features, point_features=point_features)




if __name__ == '__main__':
    from model.bacbone.pointnet2 import PointNet2
    torch.manual_seed(0)
    bz = 2
    xyz = torch.rand(bz, 2048, 3)
    backbone = PointNet2(in_channel=3)
    global_features, point_features = backbone(xyz)
    # global_features shape(bz, 1, 512) , why '1': group_all = True
    # point_features shape (bz, 64, 2048)
    head = PointCloudInstanceSegHead()
    pred_bbox, pred_bbox_score, pred_sem_seg = head(global_features=global_features,
                                                    point_features=point_features)
    pred_mask = head.forward_ins(point_features, global_features, pred_bbox, pred_bbox_score)


    losses = head.forward_train(input_pc=xyz,
                                global_features=global_features,
                                point_features=point_features,
                                gt_bbox=torch.rand(bz, 5, 2, 3),
                                gt_mask=torch.rand(bz, 1, 2048))
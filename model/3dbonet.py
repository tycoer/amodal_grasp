import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from time import time
# import point
from scipy.optimize import linear_sum_assignment


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Description:
        just the simple Euclidean distance fomula，(x-y)^2,
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def group_points(xyz,idx):
    b , n , c = xyz.shape
    m = idx.shape[1]
    nsample = idx.shape[2]
    out = torch.zeros((xyz.shape[0],xyz.shape[1], idx.shape[2],c)).cuda()
    point.group_points(b,n,c,n,nsample,xyz,idx.int(),out)
    return out

def farthest_point_sample_gpu(xyz, npoint):
    b, n ,c = xyz.shape
    centroid = torch.zeros((xyz.shape[0],npoint)).int().cuda()
    temp = torch.zeros((32,n)).cuda()
    point.farthestPoint(b,n, npoint, xyz , temp   ,centroid)
    return centroid.long()

def ball_query(radius, nsample, xyz, new_xyz):
    b, n ,c = xyz.shape
    m =  new_xyz.shape[1]
    group_idx = torch.zeros((new_xyz.shape[0],new_xyz.shape[1], nsample), dtype=torch.int32).cuda()
    pts_cnt = torch.zeros((xyz.shape[0],xyz.shape[1]), dtype=torch.int32).cuda()
    point.ball_query (b, n, m, radius, nsample, xyz, new_xyz, group_idx ,pts_cnt)
    return group_idx.long()

def idx_pts(points,idx):
    new_points = torch.cat([points.index_select(1,idx[b]) for b in range(0,idx.shape[0])], dim=0)
    return  new_points

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: the number of points that make the local region.
        radius: the radius of the local region
        nsample: the number of points in a local region
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    Np = npoint
    assert isinstance(Np, int)

    new_xyz = index_points(xyz, farthest_point_sample_gpu(xyz, npoint)) # [B,n,3] and [B,np] → [B,np,3]
    idx = ball_query(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)# [B,n,3] and [B,n,M] → [B,n,M,3]
    grouped_xyz -= new_xyz.view(B, Np, 1, C)  # the points of each group will be normalized with their centroid
    if points is not None:
        grouped_points = index_points(points, idx)# [B,n,3] and [B,n,M] → [B,n,M,3]
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Description:
        Equivalent to sample_and_group with npoint=1, radius=np.inf, and the centroid is (0, 0, 0)
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1,1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:,:,:3], idx[:,:,:3] #[B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists #[B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1) #[B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim = 2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

def gather_tensor_along_2nd_axis(bat_bb_pred, bat_bb_indices):
    bat_size, ins_max_num, d1, d2 = bat_bb_pred.shape
    bat_size_range = torch.arange(0,bat_size,1).cuda()

    bat_size_range_flat = bat_size_range.view( (-1,1))
    bat_size_range_flat_repeat = bat_size_range_flat.repeat(1,int(ins_max_num))
    bat_size_range_flat_repeat = bat_size_range_flat_repeat.view(-1).int()

    indices_2d_flat = bat_bb_indices.view(-1).int()
    indices_2d_flat_repeat = bat_size_range_flat_repeat * int(ins_max_num) + indices_2d_flat

    bat_bb_pred = bat_bb_pred.view(-1, int(d1), int(d2))
    bat_bb_pred_new = torch.index_select(bat_bb_pred,0, indices_2d_flat_repeat.long())
    bat_bb_pred_new = bat_bb_pred_new.view(bat_size, int(ins_max_num), int(d1), int(d2))

    return bat_bb_pred_new

def hungarian(cost, gt_boxes):
    box_mask = np.array([[0, 0, 0], [0, 0, 0]])
    # return ordering : batch_size x num_instances
    loss_total = 0.

    batch_size,num_instances  = cost.shape[:2]
    ordering= np.zeros(shape=[batch_size, num_instances]).astype(np.int32)
    cost_numpy = cost.clone().detach().cpu().numpy()
    gt_boxes_numpy = gt_boxes.clone().detach().cpu().numpy()
    for idx in range(batch_size):
        ins_gt_boxes = gt_boxes_numpy[idx]
        ins_count = 0
        for box in ins_gt_boxes:
            if np.array_equal(box, box_mask):
                break
            else:
                ins_count += 1
        valid_cost = cost_numpy[idx][:ins_count]
        row_ind, col_ind = linear_sum_assignment(valid_cost)

     ## ins_gt order
        unmapped = num_instances - ins_count
        if unmapped > 0:
            rest = np.array(range(ins_count, num_instances))
            row_ind = np.concatenate([row_ind, rest])
            unmapped_ind = np.array(list(set(range(num_instances)) - set(col_ind)))
            col_ind = np.concatenate([col_ind, unmapped_ind])

        loss_total += cost_numpy[idx][row_ind, col_ind].sum()
        ordering[idx] = np.reshape(col_ind, [1, -1])


    return ordering, (loss_total / float(batch_size))
    ######

def bbvert_association(xyz, y_bbvert_pred, Y_bbvert, label=''):
    points_xyz = xyz.permute(0,2,1)
    points_num = points_xyz.shape[1]
    bbnum = y_bbvert_pred.shape[1]
    gt_bbnum = Y_bbvert.shape[1]
    points_xyz = points_xyz[:,None,:, :].repeat(1,bbnum,1, 1)

    ##### get points hard mask in each gt bbox
    gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
    gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
    gt_bbox_min_xyz = gt_bbox_min_xyz[:, :,None, :].repeat(1,1,points_num, 1)
    gt_bbox_max_xyz = gt_bbox_max_xyz[:, :,None, :].repeat(1,1,points_num, 1)
    tp1_gt = gt_bbox_min_xyz - points_xyz
    tp2_gt = points_xyz - gt_bbox_max_xyz
    tp_gt = tp1_gt * tp2_gt
    points_in_gt_bbox_prob = torch.eq(torch.mean(torch.gt(tp_gt, 0.0).float(), dim=-1), 1.0).float()


    ##### get points soft mask in each pred bbox ---> Algorithm 1
    pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
    pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
    pred_bbox_min_xyz = pred_bbox_min_xyz[:,:,None, :].repeat(1, 1,points_num , 1)
    pred_bbox_max_xyz = pred_bbox_max_xyz[:,:,None, :].repeat(1, 1,points_num , 1)
    tp1_pred = pred_bbox_min_xyz - points_xyz
    tp2_pred = points_xyz - pred_bbox_max_xyz
    tp_pred = 100 * tp1_pred * tp2_pred

    tp_pred[tp_pred>=20.0] = 20.0
    tp_pred[tp_pred<=-20.0] = -20.0

    points_in_pred_bbox_prob = 1.0/(1.0 + torch.exp(-1.0 * tp_pred))
    points_in_pred_bbox_prob = torch.min(points_in_pred_bbox_prob, dim=-1)[0]


    ##### get bbox cross entropy scores
    prob_gt = points_in_gt_bbox_prob[:, :, None, :].repeat(1, 1, y_bbvert_pred.shape[1], 1)
    prob_pred = points_in_pred_bbox_prob[:, None,:, :].repeat(1, Y_bbvert.shape[1], 1, 1)

    ce_scores_matrix = - prob_gt * torch.log(prob_pred + 1e-8) - (1 - prob_gt) * torch.log(1 - prob_pred + 1e-8)
    ce_scores_matrix = torch.mean(ce_scores_matrix, dim=-1)

    ##### get bbox soft IOU
    TP = torch.sum(prob_gt * prob_pred, dim=-1)
    FP = torch.sum(prob_pred, dim=-1) - TP
    FN = torch.sum(prob_gt, dim=-1) - TP
    iou_scores_matrix = TP/ (TP + FP + FN + 1e-6)
    # iou_scores_matrix = 1.0/iou_scores_matrix  # bad, don't use
    iou_scores_matrix = -1.0 * iou_scores_matrix  # to minimize

    ##### get bbox l2 scores
    l2_gt =Y_bbvert[:, :,None , :, :].repeat(1, 1,bbnum , 1, 1)
    l2_pred = y_bbvert_pred[:, None,:, :, :].repeat(1, bbnum,1, 1, 1)
    l2_gt = l2_gt.view((-1, bbnum, bbnum, 2 * 3))
    l2_pred = l2_pred.view((-1,bbnum, bbnum, 2 * 3))
    l2_scores_matrix = torch.mean((l2_gt - l2_pred) ** 2, dim=-1)

    ##### bbox association
    if label == 'use_all_ce_l2_iou':
        associate_maxtrix = ce_scores_matrix + l2_scores_matrix + iou_scores_matrix
    elif label == 'use_both_ce_l2':
        associate_maxtrix = ce_scores_matrix + l2_scores_matrix
    elif label == 'use_both_ce_iou':
        associate_maxtrix = ce_scores_matrix + iou_scores_matrix
    elif label == 'use_both_l2_iou':
        associate_maxtrix = l2_scores_matrix + iou_scores_matrix
    elif label == 'use_only_ce':
        associate_maxtrix = ce_scores_matrix
    elif label == 'use_only_l2':
        associate_maxtrix = l2_scores_matrix
    elif label == 'use_only_iou':
        associate_maxtrix = iou_scores_matrix
    else:
        associate_maxtrix=None
        print('association label error!'); exit()


    ######
    pred_bborder, association_score_min = hungarian(associate_maxtrix, Y_bbvert)
    pred_bborder = torch.from_numpy(pred_bborder).int().cuda()

    y_bbvert_pred_new = gather_tensor_along_2nd_axis(y_bbvert_pred, pred_bborder)

    return y_bbvert_pred_new, pred_bborder

def bbscore_association(y_bbscore_pred_raw, pred_bborder):
    y_bbscore_pred_raw = y_bbscore_pred_raw.unsqueeze(2).permute(0,3,1,2)

    y_bbscore_pred_new = gather_tensor_along_2nd_axis(y_bbscore_pred_raw, pred_bborder)
    y_bbscore_pred_new = y_bbscore_pred_new.view( (-1, int(y_bbscore_pred_new.shape[1]) ))
    return y_bbscore_pred_new

def get_loss_bbvert(xyz, y_bbvert_pred, Y_bbvert, label=''):
    points_xyz = xyz.permute(0,2,1)
    points_num = points_xyz.shape[1]
    bb_num  = Y_bbvert.shape[1]
    points_xyz = points_xyz[:, None, :, :].repeat(1, bb_num, 1, 1)

    ##### get points hard mask in each gt bbox
    gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
    gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
    gt_bbox_min_xyz = gt_bbox_min_xyz[:, :, None, :].repeat(1, 1, points_num, 1)
    gt_bbox_max_xyz = gt_bbox_max_xyz[:, :, None, :].repeat(1, 1, points_num, 1)
    tp1_gt = gt_bbox_min_xyz - points_xyz
    tp2_gt = points_xyz - gt_bbox_max_xyz
    tp_gt = tp1_gt * tp2_gt
    points_in_gt_bbox_prob = torch.eq(torch.mean(torch.gt(tp_gt, 0.0).float(), dim=-1), 1.0).float()


    ##### get points soft mask in each pred bbox
    pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
    pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
    pred_bbox_min_xyz = pred_bbox_min_xyz[:, :, None, :].repeat(1, 1, points_num, 1)
    pred_bbox_max_xyz = pred_bbox_max_xyz[:, :, None, :].repeat(1, 1, points_num, 1)
    tp1_pred = pred_bbox_min_xyz - points_xyz
    tp2_pred = points_xyz - pred_bbox_max_xyz
    tp_pred = 100*tp1_pred*tp2_pred
    tp_pred[tp_pred>=20.0] = 20.0
    tp_pred[tp_pred<=-20.0] = -20.0
    points_in_pred_bbox_prob = 1.0/(1.0 + torch.exp(-1.0 * tp_pred))
    points_in_pred_bbox_prob = torch.min(points_in_pred_bbox_prob, dim=-1)[0]

    ##### helper -> the valid bbox (the gt boxes are zero-padded during data processing, pickup valid ones here)
    Y_bbox_helper = torch.sum(Y_bbvert.view( (-1, bb_num, 6)),dim=-1)
    Y_bbox_helper = torch.gt(Y_bbox_helper, 0.0).float()

    # print(points_in_gt_bbox_prob.shape,points_in_pred_bbox_prob.shape,Y_bbox_helper.shape)

    # points_in_gt_bbox_prob = points_in_gt_bbox_prob[:, :, None, :].repeat(1, 1, y_bbvert_pred.shape[1], 1)
    # points_in_pred_bbox_prob = points_in_pred_bbox_prob[:, None,:, :].repeat(1, Y_bbvert.shape[1], 1, 1)
    ##### 1. get ce loss of valid/positive bboxes, don't count the ce_loss of invalid/negative bboxes
    Y_bbox_helper_tp1 = Y_bbox_helper[:, :,None].repeat(1, 1, points_num)
    bbox_loss_ce_all = -points_in_gt_bbox_prob * torch.log(points_in_pred_bbox_prob + 1e-8) \
                   -(1.-points_in_gt_bbox_prob)*torch.log(1.-points_in_pred_bbox_prob + 1e-8)


    bbox_loss_ce_pos = torch.sum(bbox_loss_ce_all*Y_bbox_helper_tp1)/torch.sum(Y_bbox_helper_tp1)
    bbox_loss_ce = bbox_loss_ce_pos

    ##### 2. get iou loss of valid/positive bboxes
    TP = torch.sum(points_in_pred_bbox_prob * points_in_gt_bbox_prob, dim=-1)
    FP = torch.sum(points_in_pred_bbox_prob, dim=-1) - TP
    FN = torch.sum(points_in_gt_bbox_prob, dim=-1) - TP
    bbox_loss_iou_all = TP/(TP + FP + FN + 1e-6)
    bbox_loss_iou_all = -1.0*bbox_loss_iou_all

    bbox_loss_iou_pos = torch.sum(bbox_loss_iou_all*Y_bbox_helper)/torch.sum(Y_bbox_helper)
    bbox_loss_iou = bbox_loss_iou_pos

    ##### 3. get l2 loss of both valid/positive bboxes
    bbox_loss_l2_all = (Y_bbvert - y_bbvert_pred)**2
    bbox_loss_l2_all = torch.mean(bbox_loss_l2_all.view( (-1, bb_num, 6 )), dim=-1)
    bbox_loss_l2_pos = torch.sum(bbox_loss_l2_all*Y_bbox_helper)/torch.sum(Y_bbox_helper)

    ## to minimize the 3D volumn of invalid/negative bboxes, it serves as a regularizer to penalize false pred bboxes
    ## it turns out to be quite helpful, but not discussed in the paper
    bbox_pred_neg = (1.- Y_bbox_helper)[:,:,None,None].repeat(1,1,2,3)*y_bbvert_pred
    bbox_loss_l2_neg = (bbox_pred_neg[:,:,0,:]-bbox_pred_neg[:,:,1,:])**2
    bbox_loss_l2_neg = torch.sum(bbox_loss_l2_neg)/(torch.sum(1.-Y_bbox_helper)+1e-8)

    bbox_loss_l2 = bbox_loss_l2_pos + bbox_loss_l2_neg

    #####
    if label == 'use_all_ce_l2_iou':
        bbox_loss = bbox_loss_ce + bbox_loss_l2 + bbox_loss_iou
    elif label == 'use_both_ce_l2':
        bbox_loss = bbox_loss_ce + bbox_loss_l2
    elif label == 'use_both_ce_iou':
        bbox_loss = bbox_loss_ce + bbox_loss_iou
    elif label == 'use_both_l2_iou':
        bbox_loss = bbox_loss_l2 + bbox_loss_iou
    elif label == 'use_only_ce':
        bbox_loss = bbox_loss_ce
    elif label == 'use_only_l2':
        bbox_loss = bbox_loss_l2
    elif label == 'use_only_iou':
        bbox_loss = bbox_loss_iou
    else:
        bbox_loss = None
        print('bbox loss label error!'); exit()

    return bbox_loss, bbox_loss_l2, bbox_loss_ce, bbox_loss_iou

def get_loss_bbscore(y_bbscore_pred, Y_bbvert):
    bb_num =Y_bbvert.shape[1]

    ##### helper -> the valid bbox
    Y_bbox_helper = torch.sum(Y_bbvert.view((-1, bb_num, 6)), dim=-1)
    Y_bbox_helper = torch.gt(Y_bbox_helper, 0.0).float()

    ##### bbox score loss
    bbox_loss_score = torch.mean(-Y_bbox_helper * torch.log(y_bbscore_pred + 1e-8)
                                     -(1. - Y_bbox_helper) * torch.log(1. - y_bbscore_pred + 1e-8))
    return bbox_loss_score

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes,max_ins):
        super(PointNet2SemSeg, self).__init__()
        self.max_ins = max_ins
        self.sa0 = PointNetSetAbstraction(4096, 0.1, 32, 6, [16, 16, 32], False)
        self.sa05 = PointNetSetAbstraction(2048, 0.1, 32, 32+3, [32, 32, 32], False)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 32+3, [32, 32, 64], False)# npoint, radius, nsample, in_channel, mlp, group_all
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.sa5 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 512], True)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])#in_channel, mlp
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])#in_channel, mlp
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])#in_channel, mlp
        self.fp1 = PointNetFeaturePropagation(160, [128, 128, 128])#in_channel, mlp
        self.fp05 = PointNetFeaturePropagation(160, [128, 128, 64])
        self.fp0 = PointNetFeaturePropagation(67, [128, 128, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

        ##bbox

        self.fc1 = nn.Linear(in_features=512,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=256)
        self.fc3 = nn.Linear(in_features=256,out_features=256)
        self.fc4 = nn.Linear(in_features=256,out_features=max_ins*2*3)

        self.fc5 = nn.Linear(in_features=256,out_features=256)
        self.fc6 = nn.Linear(in_features=256,out_features=max_ins*1)


        self.criterion1 = nn.BCELoss().cuda()



    def forward(self, xyz,color,target_mask,target_bb):
        xyz = xyz.permute(0, 2, 1)
        color = color.permute(0, 2, 1)
        l0_xyz, l0_points = self.sa0(xyz, color)
        l05_xyz, l05_points = self.sa05(l0_xyz, l0_points)
        l1_xyz, l1_points = self.sa1(l05_xyz, l05_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l3_xyz, l3_points)


        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l05_points = self.fp1(l05_xyz, l1_xyz, l05_points, l1_points)
        l0_points = self.fp05(l0_xyz, l05_xyz, l0_points, l05_points)
        l0_points = self.fp0(xyz, l0_xyz, color, l0_points)


        global_features = l5_points.permute(0,2,1)
        point_features = l0_points

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)

        sem_seg = torch.sigmoid(x)

        ###bbox part
        b1 = F.relu(self.fc1(global_features))
        b2 = F.relu(self.fc2(b1))
                ##sub branch 1
        b3 = F.relu(self.fc3(b2))
        bbvert = self.fc4(b3)
        bbvert = bbvert.view((-1,self.max_ins ,2,3))
        points_min = torch.min(bbvert,dim=-2)[0][:, :, None, :]
        points_max = torch.max(bbvert,dim=-2)[0][:, :, None, :]
        y_bbvert_pred = torch.cat([points_min,points_max],dim=-2)
                ##sub branch 2
        b4 = F.relu(self.fc5(b2))
        y_bbscore_pred = torch.sigmoid(self.fc6(b4))


        # print(sem_seg.shape,y_bbvert_pred.shape,y_bbscore_pred.shape)
        zeros = torch.zeros((target_bb.shape[0],self.max_ins-target_bb.shape[1],2,3)).cuda()
        target_bb2 = torch.cat([target_bb,zeros],dim=1)
        ## losses
            ##loss1
        sem_loss = self.criterion1(sem_seg,target_mask)
        y_bbvert_pred_pred = y_bbvert_pred
            ##loss2
        bbox_criteria = 'use_all_ce_l2_iou'
        y_bbvert_pred, pred_bborder = bbvert_association(xyz,  y_bbvert_pred, target_bb2, label=bbox_criteria)
        y_bbscore_pred = bbscore_association(y_bbscore_pred,pred_bborder)



        bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou = get_loss_bbvert(xyz, y_bbvert_pred, target_bb2, label=bbox_criteria)
        bbscore_loss = get_loss_bbscore(y_bbscore_pred,target_bb2)
        End_Game = bbvert_loss + bbscore_loss + sem_loss
        return End_Game,y_bbvert_pred_pred,sem_seg




if __name__ == '__main__':
    for i in range(10):
        xyz = torch.rand(1, 30000,3).cuda()
        colors = torch.rand(1, 30000,3).cuda()
        mask = torch.rand(1, 1,30000).cuda()
        target_bb = torch.rand(1, 5,2,3).cuda()
        net = PointNet2SemSeg(2)
        net.cuda()
        x = net(xyz,colors,mask,target_bb)
        print(x)
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_loss_bbvert(xyz, y_bbvert_pred, Y_bbvert, label='use_all_ce_l2_iou'):
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


def get_loss_pmask(xyz, y_pmask_pred, Y_pmask):
    points_num = xyz.shape[1]
    ##### valid ins
    Y_pmask_helper = torch.sum(Y_pmask, dim=-1)
    Y_pmask_helper = torch.gt(Y_pmask_helper, 0.0).float()
    Y_pmask_helper = torch.tile(Y_pmask_helper[:, :, None], [1, 1, points_num])

    Y_pmask = Y_pmask * Y_pmask_helper
    y_pmask_pred = y_pmask_pred * Y_pmask_helper

    ##### focal loss
    alpha = 0.75
    gamma = 2
    pmask_loss_focal_all = -Y_pmask * alpha * ((1. - y_pmask_pred) ** gamma) * torch.log(y_pmask_pred + 1e-8) \
                           - (1. - Y_pmask) * (1. - alpha) * (y_pmask_pred ** gamma) * torch.log(1. - y_pmask_pred + 1e-8)
    pmask_loss_focal = torch.sum(pmask_loss_focal_all * Y_pmask_helper) / torch.sum(Y_pmask_helper)

    ## the above "alpha" makes the loss to be small
    ## then use a constant, so it's numerically comparable with other losses (e.g., semantic loss, bbox loss)
    pmask_loss = 30 * pmask_loss_focal
    return pmask_loss




def gather_tensor_along_2nd_axis(bat_bb_pred, bat_bb_indices):
    bat_size, ins_max_num, d1, d2 = bat_bb_pred.shape
    bat_size_range = torch.arange(0,bat_size,1)

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

def bbvert_association(xyz, y_bbvert_pred, Y_bbvert, label='use_all_ce_l2_iou'):
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
    pred_bborder = torch.from_numpy(pred_bborder).int()

    y_bbvert_pred_new = gather_tensor_along_2nd_axis(y_bbvert_pred, pred_bborder)

    return y_bbvert_pred_new, pred_bborder

def bbscore_association(y_bbscore_pred_raw, pred_bborder):
    y_bbscore_pred_raw = y_bbscore_pred_raw.unsqueeze(2).permute(0,3,1,2)

    y_bbscore_pred_new = gather_tensor_along_2nd_axis(y_bbscore_pred_raw, pred_bborder)
    y_bbscore_pred_new = y_bbscore_pred_new.view( (-1, int(y_bbscore_pred_new.shape[1]) ))
    return y_bbscore_pred_new




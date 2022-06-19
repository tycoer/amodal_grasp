import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import HEADS


def grasp_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_grasps_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    grasp_targets = map(grasp_target_single, pos_proposals_list,
                        pos_assigned_gt_inds_list, gt_grasps_list, cfg_list)
    grasp_targets = torch.cat(list(grasp_targets))
    return grasp_targets


def grasp_target_single(pos_proposals, pos_assigned_gt_inds, gt_grasps, cfg):
    grasp_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    grasp_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_grasp = gt_grasps[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = cv2.resize(gt_grasp[y1:y1 + h, x1:x1 + w],
                                dsize=(grasp_size, grasp_size),
                                interpolation=cv2.INTER_NEAREST)
            grasp_targets.append(target)
        grasp_targets = torch.from_numpy(np.stack(grasp_targets)).float().to(
            pos_proposals.device)
    else:
        grasp_targets = pos_proposals.new_zeros((0, grasp_size, grasp_size, 3))
    return grasp_targets


@HEADS.register_module()
class GraspHead(torch.nn.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 num_bins=-1,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_qual_weight=1,
                 loss_quat_weight=3,
                 loss_width_weight=1,
                 loss_depth_weight=1,

                 ):
        super().__init__()

        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.loss_qual_weight = loss_qual_weight
        self.loss_quat_weight = loss_quat_weight
        self.loss_width_weight = loss_width_weight
        self.loss_depth_weight = loss_depth_weight


        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)

        self.convs_qual = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_qual.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs_quat = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_quat.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs_width = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_width.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs_depth = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs_depth.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)

        self.conv_logits_qual = nn.Conv2d(logits_in_channel, 1, 1)
        self.conv_logits_quat = nn.Conv2d(logits_in_channel, 4, 1)
        self.conv_logits_width = nn.Conv2d(logits_in_channel, 1, 1)
        self.conv_logits_depth = nn.Conv2d(logits_in_channel, 1, 1)

    def get_target(self, sampling_results, gt_grasps, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        grasp_targets = grasp_target(pos_proposals, pos_assigned_gt_inds,
                                     gt_grasps, rcnn_train_cfg)
        return grasp_targets

    @auto_fp16()
    def forward(self, x, **kwargs):
        for i in range(self.num_convs):
            if i == 0:
                qual = self.convs_qual[i](x)
                quat = self.convs_quat[i](x)
                width = self.convs_width[i](x)
                depth = self.convs_depth[i](x)
            else:
                qual = self.convs_qual[i](qual)
                quat = self.convs_quat[i](quat)
                width = self.convs_width[i](width)
                depth = self.convs_depth[i](depth)

        qual = self.conv_logits_qual(qual)
        quat = self.conv_logits_quat(quat)
        width = self.conv_logits_width(width)
        depth = self.conv_logits_depth(depth)
        return qual, quat, width, depth

    def loss(self,
             qual_pred,
             quat_pred,
             width_pred,
             depth_pred,
             grasp_targets):

        qual_target, quat_target, width_target, depth_target, tolerance_target = (grasp_targets[:, :, :, 0],
                                                                        grasp_targets[:, :, :, 1:5],
                                                                        grasp_targets[:, :, :, 5],
                                                                        grasp_targets[:, :, :, 6],
                                                                        grasp_targets[:, :, :, 7],
                                                                        )

        loss_qual = sigmoid_focal_loss(qual_pred.squeeze(1), qual_target, reduction='none')
        loss_quat = F.mse_loss(quat_pred.permute(0, 2, 3, 1), quat_target, reduction='none')
        loss_width = F.mse_loss(width_pred.squeeze(1), width_target, reduction='none')
        loss_depth = F.mse_loss(depth_pred.squeeze(1), depth_target, reduction='none')

        # 只计算正样本的loss
        loss_qual = (qual_target * loss_qual).mean() * self.loss_qual_weight
        loss_quat = (qual_target.unsqueeze(-1) * loss_quat).mean() * self.loss_quat_weight
        loss_width = (qual_target * loss_width).mean() * self.loss_width_weight
        loss_depth = (qual_target * loss_depth).mean() * self.loss_depth_weight


        losses = dict(loss_qual=loss_qual,
                      loss_width=loss_width,
                      loss_quat=loss_quat,
                      loss_depth=loss_depth,)
        return losses

    def get_nocs(self, grasp_pred, det_bboxes, det_labels, rcnn_test_cfg,
                 ori_shape, scale_factor, window, rescale):

        grasp_pred = torch.cat(grasp_pred)
        cls_nocs = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
            shift = window[:2]
            shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0
            shifts = np.array([0, 0, 0, 0])

        for i in range(bboxes.shape[0]):
            # bbox = ((bboxes[i, :] - shifts) / scale_factor).astype(np.int32)
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)
            grasp_pred_ = grasp_pred[i, :, :, 0, :]
            im_nocs = np.zeros((img_h, img_w, 3), dtype=np.float32)

            bbox_nocs = cv2.resize(grasp_pred_,
                                   dsize=(w, h),
                                   interpolation=cv2.INTER_NEAREST)
            im_nocs[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w, :] = bbox_nocs
            cls_nocs[label - 1].append(im_nocs)

        return cls_nocs
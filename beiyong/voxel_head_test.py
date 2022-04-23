# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from mmcv.cnn import ConvModule
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
# from detectron2.utils.events import get_event_storage
# from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F
from .utils.voxel import batch_crop_voxels_within_box
import numpy as np
from mmdet.models.builder import HEADS
from pytorch3d.ops import cubify



def cat(tensors:list, dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def voxel_rcnn_loss(pred_voxel_logits, instances, loss_weight=1.0):
    """
    Compute the voxel prediction loss defined in the Mesh R-CNN paper.

    Args:
        pred_voxel_logits (Tensor): A tensor of shape (B, C, D, H, W) or (B, 1, D, H, W)
            for class-specific or class-agnostic, where B is the total number of predicted voxels
            in all images, C is the number of foreground classes, and D, H, W are the depth,
            height and width of the voxel predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_voxel_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        loss_weight (float): A float to multiply the loss with.

    Returns:
        voxel_loss (Tensor): A scalar tensor containing the loss.
    """
    reg_class_agnostic = pred_voxel_logits.size(1) == 1
    total_num_voxels = pred_voxel_logits.size(0)
    voxel_side_len = pred_voxel_logits.size(2)
    assert pred_voxel_logits.size(2) == pred_voxel_logits.size(
        3
    ), "Voxel prediction must be square!"
    assert pred_voxel_logits.size(2) == pred_voxel_logits.size(
        4
    ), "Voxel prediction must be square!"

    gt_classes = []
    gt_voxel_logits = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not reg_class_agnostic:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_voxels = instances_per_image.gt_voxels
        gt_K = instances_per_image.gt_K
        gt_voxel_logits_per_image = batch_crop_voxels_within_box(
            gt_voxels, instances_per_image.proposal_boxes.tensor, gt_K, voxel_side_len
        ).to(device=pred_voxel_logits.device)
        gt_voxel_logits.append(gt_voxel_logits_per_image)

    if len(gt_voxel_logits) == 0:
        return pred_voxel_logits.sum() * 0, gt_voxel_logits

    gt_voxel_logits = cat(gt_voxel_logits, dim=0)
    assert gt_voxel_logits.numel() > 0, gt_voxel_logits.shape

    if reg_class_agnostic:
        pred_voxel_logits = pred_voxel_logits[:, 0]
    else:
        indices = torch.arange(total_num_voxels)
        gt_classes = cat(gt_classes, dim=0)
        pred_voxel_logits = pred_voxel_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    # Note that here we allow gt_voxel_logits to be float as well
    # (depend on the implementation of rasterize())
    voxel_accurate = (pred_voxel_logits > 0.5) == (gt_voxel_logits > 0.5)
    voxel_accuracy = voxel_accurate.nonzero().size(0) / voxel_accurate.numel()

    voxel_loss = F.binary_cross_entropy_with_logits(
        pred_voxel_logits, gt_voxel_logits, reduction="mean"
    )
    voxel_loss = voxel_loss * loss_weight
    return voxel_loss, gt_voxel_logits


def voxel_rcnn_inference(pred_voxel_logits, pred_instances):
    """
    Convert pred_voxel_logits to estimated foreground probability voxels while also
    extracting only the voxels for the predicted classes in pred_instances. For each
    predicted box, the voxel of the same class is attached to the instance by adding a
    new "pred_voxels" field to pred_instances.

    Args:
        pred_voxel_logits (Tensor): A tensor of shape (B, C, D, H, W) or (B, 1, D, H, W)
            for class-specific or class-agnostic, where B is the total number of predicted voxels
            in all images, C is the number of foreground classes, and D, H, W are the depth, height
            and width of the voxel predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_voxels" field storing a voxel of size (D, H,
            W) for predicted class. Note that the voxels are returned as a soft (non-quantized)
            voxels the resolution predicted by the network; post-processing steps are left
            to the caller.
    """
    reg_class_agnostic = pred_voxel_logits.size(1) == 1

    if reg_class_agnostic:
        voxel_probs_pred = pred_voxel_logits.sigmoid()
    else:
        # Select voxels corresponding to the predicted classes
        num_voxels = pred_voxel_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_voxels, device=class_pred.device)
        voxel_probs_pred = pred_voxel_logits[indices, class_pred][:, None].sigmoid()
    # voxel_probs_pred.shape: (B, 1, D, H, W)

    num_boxes_per_image = [len(i) for i in pred_instances]
    voxel_probs_pred = voxel_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(voxel_probs_pred, pred_instances):
        instances.pred_voxels = prob  # (1, D, H, W)


@HEADS.register_module()
class VoxelRCNNConvUpsampleHead(nn.Module):
    """
    A voxel head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """
    def __init__(self,
                 num_classes=9,
                 input_channels=256,
                 num_conv=256,
                 conv_dims=256,
                 num_depth=28,
                 reg_class_agnostic=True,
                 loss_weight=1,
                 cubify_thresh=0.2
                 ):


    # def __init__(self, cfg, input_shape):
        super(VoxelRCNNConvUpsampleHead, self).__init__()
        self.conv_norm_relus = []
        self.num_depth = num_depth
        self.num_classes = 1 if reg_class_agnostic else num_classes
        self.norm  = ''
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("voxel_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = nn.ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.predictor = Conv2d(
            conv_dims, self.num_classes * self.num_depth, kernel_size=1, stride=1, padding=0
        )

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for voxel prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        self.loss_weight = loss_weight
        self.cubify_thresh = cubify_thresh

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        x = self.predictor(x)
        # reshape from (N, CD, H, W) to (N, C, D, H, W)
        x = x.reshape(x.size(0), self.num_classes, self.num_depth, x.size(2), x.size(3))
        return x

    def loss(self,
             pred_voxel_logits,
             sampling_results,
             gt_voxels,
             Ks,
             ):
        voxel_side_len = pred_voxel_logits.size(2)
        ##########################  get_targets (namely the gt_voxel_logits) ######################
        gt_voxel_logits = []
        for sampling_result, voxel, K in zip(sampling_results, gt_voxels, Ks):
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            pos_proposals = sampling_result.pos_bboxes

            Ks_sampled = [K[i] for i in pos_assigned_gt_inds]
            gt_voxels_sampled = [voxel[i] for i in pos_assigned_gt_inds]

            voxel_target = batch_crop_voxels_within_box(voxels=gt_voxels_sampled,
                                                        boxes=pos_proposals,
                                                        Ks=Ks_sampled,
                                                        voxel_side_len=voxel_side_len)
            gt_voxel_logits.append(voxel_target)

        ##############################################################
        if len(gt_voxel_logits) == 0:
            return pred_voxel_logits.sum() * 0, gt_voxel_logits

        gt_voxel_logits = cat(gt_voxel_logits, dim=0)
        assert gt_voxel_logits.numel() > 0, gt_voxel_logits.shape
        pred_voxel_logits = pred_voxel_logits[:, 0]

        # if reg_class_agnostic:
        #     pred_voxel_logits = pred_voxel_logits[:, 0]
        # else:
        #     indices = torch.arange(total_num_voxels)
        #     gt_classes = cat(gt_classes, dim=0)
        #     pred_voxel_logits = pred_voxel_logits[indices, gt_classes]

        # Log the training accuracy (using gt classes and 0.5 threshold)
        # Note that here we allow gt_voxel_logits to be float as well
        # (depend on the implementation of rasterize())
        # voxel_accurate = (pred_voxel_logits > 0.5) == (gt_voxel_logits > 0.5)
        # voxel_accuracy = voxel_accurate.nonzero().size(0) / voxel_accurate.numel()
        # voxel_accuracy: float
        voxel_loss = F.binary_cross_entropy_with_logits(
            pred_voxel_logits, gt_voxel_logits, reduction="mean"
        )
        voxel_loss = self.loss_weight * voxel_loss
        # print(voxel_loss, voxel_accuracy)
        return voxel_loss, gt_voxel_logits \
               # ,voxel_accuracy

    def init_mesh(self,
                  voxel_pred,
                  ):
        # with torch.no_grad():
        vox_in = voxel_pred.sigmoid().squeeze(1)  # (N, V, V, V)
        init_mesh = cubify(vox_in, self.cubify_thresh)  # 1
        return init_mesh
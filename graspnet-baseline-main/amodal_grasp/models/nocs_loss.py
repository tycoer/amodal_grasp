import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.builder import LOSSES


EPSILON = 1e-7


def one_hot(indices, depth):
    inds = indices.clone()
    inds[inds < 0] = depth
    inds[inds >= depth] = depth

    output = F.one_hot(inds, depth + 1)
    output = output[..., :-1]
    # TODO: need optimize
    #output = F.one_hot(indices.clamp(min=0, max=depth-1), depth)

    #exc_inds = indices >= depth
    #exc_inds = torch.nonzero(exc_inds)
    #for exc_ind in exc_inds:
        #output[list(exc_ind)] = 0

    #neg_inds = indices < 0
    #neg_inds = torch.nonzero(neg_inds)
    #for neg_ind in neg_inds:
        #output[list(neg_ind)] = 0
    del inds

    return output


def label_to_theta(labels):
    indexes = torch.zeros_like(labels).to(labels.device)
    for i in [1, 2, 4]:
        indexes += (labels == i).long()
    indexes = torch.nonzero(indexes)[:, 0].to(labels.device)
    theta = torch.zeros_like(labels, dtype=torch.float32).to(labels.device)
    theta[indexes] = torch.Tensor([2 * math.pi / 6]).to(labels.device)

    return theta


def rotation_y_matrix(labels_rotation_theta):
    rotation_matrixes = []
    for theta in labels_rotation_theta:
        rotation_matrixes.append(torch.Tensor([
            [theta.cos(),    0.,   theta.sin()],
            [0.,             1.,   0.],
            [-theta.sin(),   0.,   theta.cos()]
        ]).unsqueeze(0).to(theta.device))

    return torch.cat(rotation_matrixes)



@LOSSES.register_module()
class SymmetryCoordLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SymmetryCoordLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                coords,
                coords_targets,
                labels,
                mask_targets,
                use_bins=False):
        coords = torch.cat([coords[0].unsqueeze(-1), coords[1].unsqueeze(-1), coords[2].unsqueeze(-1)], -1)
        if use_bins:
            coords = coords.permute(0, 3, 1, 2, 5, 4)
            num_rois, num_classes, roi_height, roi_width, _, num_bins = coords.shape
        else:
            num_rois, num_classes, roi_height, roi_width, _ = coords.shape

        labels_rotation_theta = label_to_theta(labels)
        labels_rotation_matrix = rotation_y_matrix(labels_rotation_theta)
        labels_rotation_matrix = labels_rotation_matrix.unsqueeze(1).unsqueeze(2).repeat((1, roi_height, roi_width, 1, 1))

        coords_targets -= 0.5
        coords_targets = coords_targets.unsqueeze(-1)

        rotated_y_true_1 = torch.matmul(labels_rotation_matrix, coords_targets)
        rotated_y_true_2 = torch.matmul(labels_rotation_matrix, rotated_y_true_1)
        rotated_y_true_3 = torch.matmul(labels_rotation_matrix, rotated_y_true_2)
        rotated_y_true_4 = torch.matmul(labels_rotation_matrix, rotated_y_true_3)
        rotated_y_true_5 = torch.matmul(labels_rotation_matrix, rotated_y_true_4)
        y_true_stack = torch.cat([coords_targets,
                                  rotated_y_true_1,
                                  rotated_y_true_2,
                                  rotated_y_true_3,
                                  rotated_y_true_4,
                                  rotated_y_true_5], 4)
        y_true_stack = y_true_stack.permute(0, 1, 2, 4, 3)
        y_true_stack += 0.5

        coords = coords[torch.Tensor(range(num_rois)).long(), labels]

        if use_bins:
            y_true_bins_stack = y_true_stack * num_bins - 0.000001
            y_true_bins_stack = torch.floor(y_true_bins_stack).long()
            y_true_bins_stack = one_hot(y_true_bins_stack, num_bins)

            coords = coords.unsqueeze(3).repeat(1, 1, 1, y_true_stack.shape[3], 1, 1)

            # cross entropy
            y_true_bins_stack = y_true_bins_stack.type_as(coords)
            y_true_bins_stack = y_true_bins_stack / y_true_bins_stack.sum(-1).unsqueeze(-1).clamp(min=EPSILON)
            loss = -1 * (y_true_bins_stack * coords.log()).sum(-1).clamp(max=1 - EPSILON)

        else:
            coords = coords.unsqueeze(3).repeat(1, 1, 1, y_true_stack.shape[3], 1)
            beta = 0.1
            loss = torch.abs(coords - y_true_stack)
            loss = torch.where(loss < beta, 0.5 * loss * loss / beta,
                               loss - 0.5 * beta)

        reshaped_mask = mask_targets.view(mask_targets.shape[0], mask_targets.shape[1], mask_targets.shape[2], 1, 1)
        num_of_pixels = mask_targets.sum([1, 2]) + 0.00001
        loss_in_mask = (loss * reshaped_mask).sum([1, 2])
        total_loss_in_mask = loss_in_mask.sum(-1)

        arg_min_rotation = total_loss_in_mask.argmin(-1).long()
        min_loss_in_mask = loss_in_mask[torch.Tensor(range(num_rois)).long(), arg_min_rotation]
        mean_loss_in_mask = min_loss_in_mask / num_of_pixels.unsqueeze(-1)
        loss_nocs = mean_loss_in_mask.mean(dim=0)

        return loss_nocs





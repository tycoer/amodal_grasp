# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from mmdet.models.detectors import TwoStageDetector
from .shape import box2D_to_cuboid3D, cuboid3D_to_unitbox3D
# import open3d as o3d

def test_cubify(voxels, cropped_verts):
    cropped_v = cropped_verts[0].detach().cpu().numpy()
    cropped_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cropped_v))
    o3d.io.write_point_cloud('/home/hanyang/cropped_vox.ply', cropped_pc)

    v = voxels[0].detach().cpu().numpy()
    v_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
    o3d.io.write_point_cloud('/home/hanyang/vox.ply', v_pc)


def batch_crop_voxels_within_box(voxels, boxes, Ks, voxel_side_len):
    """
    Batched version of :func:`crop_voxel_within_box`.

    Args:
        voxels (VoxelInstances): store N voxels for an image
        boxes (Tensor): store N boxes corresponding to the masks.
        Ks (Tensor): store N camera matrices
        voxel_side_len (int): the size of the voxel.

    Returns:
        Tensor: A byte tensor of shape (N, voxel_side_len, voxel_side_len, voxel_side_len),
        where N is the number of predicted boxes for this image.
    """
    device = boxes.device
    results = []
    for i in range(len(voxels)):
        # tycoer
        voxel = voxels[i].to(device)
        zrange = torch.tensor([[voxel[:, 2].min(), voxel[:, 2].max()]], device=device)
        K = torch.atleast_2d(Ks[i])
        box = torch.atleast_2d(boxes[i])
        im_sizes = torch.atleast_2d(Ks[i, 1:] * 2)
        cub3D = box2D_to_cuboid3D(zrange, K, box.clone(), im_sizes)
        txz, tyz = cuboid3D_to_unitbox3D(cub3D)
        x, y, z = voxel.split(1, dim=1)  # (num_points, 1)
        xz = torch.cat([x, z], dim=1).unsqueeze(0) # (1, num_points, 2)
        yz = torch.cat([y, z], dim=1).unsqueeze(0) # (1, num_points, 2)
        pxz = txz(xz)
        pyz = tyz(yz)
        cropped_verts = torch.stack([pxz[:, :, 0], pyz[:, :, 0], pxz[:, :, 1]], dim=2)
        results.append(verts2voxel(cropped_verts[0], [voxel_side_len] * 3))

    if len(results) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(results, dim=0).to(device=device)

def voxel_rcnn_loss(pred_voxel_logits, gt_voxels, Ks, loss_weight=1.0):
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
    cls_agnostic_voxel = pred_voxel_logits.size(1) == 1
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
        if not cls_agnostic_voxel:
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

    gt_voxel_logits = torch.cat(gt_voxel_logits, dim=0)
    assert gt_voxel_logits.numel() > 0, gt_voxel_logits.shape

    if cls_agnostic_voxel:
        pred_voxel_logits = pred_voxel_logits[:, 0]
    else:
        indices = torch.arange(total_num_voxels)
        gt_classes = torch.cat(gt_classes, dim=0)
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

class VoxelInstances:
    """
    Class to hold voxels of varying dimensions to interface with Instances
    """

    def __init__(self, voxels):
        assert isinstance(voxels, list)
        assert torch.is_tensor(voxels[0])
        self.data = voxels

    def to(self, device):
        to_voxels = [voxel.to(device) for voxel in self]
        return VoxelInstances(to_voxels)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_data = [self.data[item]]
        else:
            # advanced indexing on a single dimension
            selected_data = []
            if isinstance(item, torch.Tensor) and (
                item.dtype == torch.uint8 or item.dtype == torch.bool
            ):
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_data.append(self.data[i])
        return VoxelInstances(selected_data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}) ".format(len(self))
        return s


def downsample(vox_in, n, use_max=True):
    """
    Downsample a 3-d tensor n times
    Inputs:
      - vox_in (Tensor): HxWxD tensor
      - n (int): number of times to downsample each dimension
      - use_max (bool): use maximum value when downsampling. If set to False
                        the mean value is used.
    Output:
      - vox_out (Tensor): (H/n)x(W/n)x(D/n) tensor
    """
    dimy = vox_in.size(0) // n
    dimx = vox_in.size(1) // n
    dimz = vox_in.size(2) // n
    vox_out = torch.zeros((dimy, dimx, dimz))
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                subx = x * n
                suby = y * n
                subz = z * n
                subvox = vox_in[suby : suby + n, subx : subx + n, subz : subz + n]
                if use_max:
                    vox_out[y, x, z] = torch.max(subvox)
                else:
                    vox_out[y, x, z] = torch.mean(subvox)
    return vox_out


def verts2voxel(verts, voxel_size):
    def valid_coords(x, y, z, vx_size):
        Hv, Wv, Zv = vx_size
        indx = (x >= 0) * (x < Wv)
        indy = (y >= 0) * (y < Hv)
        indz = (z >= 0) * (z < Zv)
        return indx * indy * indz

    Hv, Wv, Zv = voxel_size
    # create original voxel of size VxVxV
    orig_voxel = torch.zeros((Hv, Wv, Zv), dtype=torch.float32)

    x = (verts[:, 0] + 1) * (Wv - 1) / 2
    x = x.long()
    y = (verts[:, 1] + 1) * (Hv - 1) / 2
    y = y.long()
    z = (verts[:, 2] + 1) * (Zv - 1) / 2
    z = z.long()

    keep = valid_coords(x, y, z, voxel_size)
    x = x[keep]
    y = y[keep]
    z = z[keep]

    orig_voxel[y, x, z] = 1.0

    # align with image coordinate system
    flip_idx = torch.tensor(list(range(Hv)[::-1]))
    orig_voxel = orig_voxel.index_select(0, flip_idx)
    flip_idx = torch.tensor(list(range(Wv)[::-1]))
    orig_voxel = orig_voxel.index_select(1, flip_idx)
    return orig_voxel


def normalize_verts(verts):
    # centering and normalization
    min, _ = torch.min(verts, 0)
    min_x, min_y, min_z = min
    max, _ = torch.max(verts, 0)
    max_x, max_y, max_z = max
    x_ctr = (min_x + max_x) / 2.0
    y_ctr = (min_y + max_y) / 2.0
    z_ctr = (min_z + max_z) / 2.0
    x_scale = 2.0 / (max_x - min_x)
    y_scale = 2.0 / (max_y - min_y)
    z_scale = 2.0 / (max_z - min_z)
    verts[:, 0] = (verts[:, 0] - x_ctr) * x_scale
    verts[:, 1] = (verts[:, 1] - y_ctr) * y_scale
    verts[:, 2] = (verts[:, 2] - z_ctr) * z_scale
    return verts

import torch
import numpy as np
import mmcv


def coord_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_coords_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    coord_targets = map(coord_target_single, pos_proposals_list,
                        pos_assigned_gt_inds_list, gt_coords_list, cfg_list)
    coord_targets = torch.cat(list(coord_targets))
    return coord_targets


def coord_target_single(pos_proposals, pos_assigned_gt_inds, gt_coords, cfg):
    coord_size = cfg.mask_size
    num_pos = pos_proposals.size(0)
    coord_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_coord = gt_coords[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            target = mmcv.imresize(gt_coord[y1:y1 + h, x1:x1 + w],
                                   (coord_size, coord_size))
            coord_targets.append(target)
        coord_targets = torch.from_numpy(np.stack(coord_targets)).float().to(
            pos_proposals.device)
    else:
        coord_targets = pos_proposals.new_zeros((0, coord_size, coord_size, 3))
    return coord_targets



import torch.nn as nn
import torch
import numpy as np

class MirrorMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, prediction, target):
        add_vec = torch.torch.tensor(
            [0.5, 0, 0], dtype=target.dtype, device=target.device)
        mult_vec = torch.tensor(
            [-1, 1, 1], dtype=target.dtype, device=target.device)
        target_mirror = (target - add_vec) * mult_vec + add_vec

        reg_loss = self.mse(prediction, target)
        mirror_loss = self.mse(prediction, target_mirror)
        loss = torch.min(reg_loss, mirror_loss)

        return loss


class VirtualGrid:
    def __init__(self,
                 lower_corner=(0, 0, 0),
                 upper_corner=(1, 1, 1),
                 grid_shape=(32, 32, 32),
                 batch_size=8,
                 device=torch.device('cpu'),
                 int_dtype=torch.int64,
                 float_dtype=torch.float32,
                 ):
        self.lower_corner = tuple(lower_corner)
        self.upper_corner = tuple(upper_corner)
        self.grid_shape = tuple(grid_shape)
        self.batch_size = int(batch_size)
        self.device = device
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

    @property
    def num_grids(self):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        return int(np.prod((batch_size,) + grid_shape))

    def get_grid_idxs(self, include_batch=True):
        batch_size = self.batch_size
        grid_shape = self.grid_shape
        device = self.device
        int_dtype = self.int_dtype
        dims = grid_shape
        if include_batch:
            dims = (batch_size,) + grid_shape
        axis_coords = [torch.arange(0, x, device=device, dtype=int_dtype)
                       for x in dims]
        coords_per_axis = torch.meshgrid(*axis_coords)
        grid_idxs = torch.stack(coords_per_axis, axis=-1)
        return grid_idxs

    def get_grid_points(self, include_batch=True):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        device = self.device
        grid_idxs = self.get_grid_idxs(include_batch=include_batch)

        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape,
                                 dtype=float_dtype, device=device) - 1
        scales = (uc - lc) / idx_scale
        offsets = -lc

        grid_idxs_no_batch = grid_idxs
        if include_batch:
            grid_idxs_no_batch = grid_idxs[:, :, :, :, 1:]
        grid_idxs_f = grid_idxs_no_batch.to(float_dtype)
        grid_points = grid_idxs_f * scales + offsets
        return grid_points

    def get_points_grid_idxs(self, points, batch_idx=None):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        int_dtype = self.int_dtype
        float_dtype = self.float_dtype
        device = self.device
        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape,
                                 dtype=float_dtype, device=device) - 1
        offsets = -lc
        scales = idx_scale / (uc - lc)
        points_idxs_f = (points + offsets) * scales
        points_idxs_i = points_idxs_f.to(dtype=int_dtype)
        points_idxs = torch.empty_like(points_idxs_i)
        for i in range(3):
            points_idxs[..., i] = torch.clamp(
                points_idxs_i[..., i], min=0, max=grid_shape[i] - 1)
        final_points_idxs = points_idxs
        if batch_idx is not None:
            final_points_idxs = torch.cat(
                [batch_idx.view(*points.shape[:-1], 1).to(
                    dtype=points_idxs.dtype), points_idxs],
                axis=-1)
        return final_points_idxs

    def flatten_idxs(self, idxs, keepdim=False):
        grid_shape = self.grid_shape
        batch_size = self.batch_size

        coord_size = idxs.shape[-1]
        target_shape = None
        if coord_size == 4:
            # with batch
            target_shape = (batch_size,) + grid_shape
        elif coord_size == 3:
            # without batch
            target_shape = grid_shape
        else:
            raise RuntimeError("Invalid shape {}".format(str(idxs.shape)))
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)
        flat_idxs = (idxs * torch.tensor(target_stride,
                                         dtype=idxs.dtype, device=idxs.device)).sum(
            axis=-1, keepdim=keepdim, dtype=idxs.dtype)
        return flat_idxs

    def unflatten_idxs(self, flat_idxs, include_batch=True):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        target_shape = grid_shape
        if include_batch:
            target_shape = (batch_size,) + grid_shape
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)

        source_shape = tuple(flat_idxs.shape)
        if source_shape[-1] == 1:
            source_shape = source_shape[:-1]
            flat_idxs = flat_idxs[..., 0]
        source_shape += (4,) if include_batch else (3,)

        idxs = torch.empty(size=source_shape,
                           dtype=flat_idxs.dtype, device=flat_idxs.device)
        mod = flat_idxs
        for i in range(source_shape[-1]):
            idxs[..., i] = mod / target_stride[i]
            mod = mod % target_stride[i]
        return idxs

    def idxs_to_points(self, idxs):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        int_dtype = idxs.dtype
        device = idxs.device

        source_shape = idxs.shape
        point_idxs = None
        if source_shape[-1] == 4:
            # has batch idx
            point_idxs = idxs[..., 1:]
        elif source_shape[-1] == 3:
            point_idxs = idxs
        else:
            raise RuntimeError("Invalid shape {}".format(tuple(source_shape)))

        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape,
                                 dtype=float_dtype, device=device) - 1
        offsets = lc
        scales = (uc - lc) / idx_scale

        idxs_points = point_idxs * scales + offsets
        return idxs_points
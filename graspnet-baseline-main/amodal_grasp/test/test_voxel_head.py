from amodal_grasp.models.voxel_head import VoxelRCNNConvUpsampleHead
import torch


if __name__ == '__main__':
    m = VoxelRCNNConvUpsampleHead()
    t = torch.rand(5, 256, 12, 12)
    res = m(t)
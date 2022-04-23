import torch.nn as nn
from .pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation
from mmdet.models.builder import BACKBONES

@BACKBONES.register_module()
class PointNet2(nn.Module):
    def __init__(self, in_channel=3):
        super(PointNet2, self).__init__()
        # npoint, radius, nsample, in_channel, mlp, group_all
        self.sa0 = PointNetSetAbstraction(4096, 0.1, 32, in_channel, [16, 16, 32], False)
        self.sa05 = PointNetSetAbstraction(2048, 0.1, 32, 32+3, [32, 32, 32], False)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 32+3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.sa5 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 512], True)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])#in_channel, mlp
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])#in_channel, mlp
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])#in_channel, mlp
        self.fp1 = PointNetFeaturePropagation(160, [128, 128, 128])#in_channel, mlp
        self.fp05 = PointNetFeaturePropagation(160, [128, 128, 64])
        self.fp0 = PointNetFeaturePropagation(64, [128, 128, 64])

    def forward(self, xyz, color=None):
        xyz = xyz.permute(0, 2, 1)
        if color is not None:
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
        return global_features, point_features


if __name__ == '__main__':
    pass

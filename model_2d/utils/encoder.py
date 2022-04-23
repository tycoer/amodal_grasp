import torch
import torch.nn as nn
from .utils import *
import torch.nn.functional as F
from torch_scatter import scatter_mean



class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self,
                 dim=1,
                 c_dim=32,
                 unet=False,
                 unet_kwargs=None,
                 unet3d=False,
                 unet3d_kwargs=None,
                 plane_resolution=512,
                 grid_resolution=None,
                 plane_type='xz',
                 kernel_size=3,
                 padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(dim, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(dim, c_dim, kernel_size, padding=1)

        # if unet:
        #     self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        # else:
        #     self.unet = None
        #
        # if unet3d:
        #     self.unet3d = UNet3D(**unet3d_kwargs)
        # else:
        #     self.unet3d = None


        self.unet = None
        self.unet3d=None


        self.c_dim = c_dim

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution

        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def forward(self, x):  # (32, 2, 40, 40, 40)
        batch_size = x.size(0)
        device = x.device
        # n_voxel = x.size(1) * x.size(2) * x.size(3)
        n_voxel = x.size(2) * x.size(3) * x.size(4)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(4)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)  # (32, 40, 40, 40, 1)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)  # (32, 40, 40, 40, 1)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)  # (32, 40, 40, 40, 1)
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)  # (32, 64000, 3)

        # Acquire voxel-wise feature
        # x = x.unsqueeze(1)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)  # (32, 32, 64000)
        c = c.permute(0, 2, 1)  # (32, 64000, 32)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea
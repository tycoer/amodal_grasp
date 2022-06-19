# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from pytorch3d.structures import Meshes
from .shape import box2D_to_cuboid3D, cuboid3D_to_unitbox3D


def batch_crop_meshes_within_box(meshes, boxes, Ks):
    device = boxes.device
    verts_list = []
    faces_list = []
    for i in range(len(meshes)):
        # tycoer
        verts, faces = meshes[i]
        verts, faces = verts.to(device), faces.to(device)
        zrange = torch.tensor([[verts[:, 2].min(), verts[:, 2].max()]], device=device)
        K = torch.atleast_2d(Ks[i])
        box = torch.atleast_2d(boxes[i])
        im_sizes = torch.atleast_2d(Ks[i, 1:] * 2)
        cub3D = box2D_to_cuboid3D(zrange, K, box.clone(), im_sizes)
        txz, tyz = cuboid3D_to_unitbox3D(cub3D)
        x, y, z = verts.split(1, dim=1)  # (num_points, 1)
        xz = torch.cat([x, z], dim=1).unsqueeze(0) # (1, num_points, 2)
        yz = torch.cat([y, z], dim=1).unsqueeze(0) # (1, num_points, 2)
        pxz = txz(xz)
        pyz = tyz(yz)
        new_verts = torch.stack([pxz[:, :, 0], pyz[:, :, 0], pxz[:, :, 1]], dim=2)

        verts_list.append(new_verts[0])
        faces_list.append(faces)
    return Meshes(verts=verts_list, faces=faces_list).to(device=device)


class MeshInstances:
    """
    Class to hold meshes of varying topology to interface with Instances
    """

    def __init__(self, meshes):
        assert isinstance(meshes, list)
        assert torch.is_tensor(meshes[0][0])
        assert torch.is_tensor(meshes[0][1])
        self.data = meshes

    def to(self, device):
        to_meshes = [(mesh[0].to(device), mesh[1].to(device)) for mesh in self]
        return MeshInstances(to_meshes)

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
        return MeshInstances(selected_data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}) ".format(len(self))
        return s

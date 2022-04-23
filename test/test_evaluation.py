from mmdet.models.builder import build_detector
from mmcv.utils import Config
from mmdet.datasets.builder import build_dataloader, build_dataset
from mmcv.runner import load_checkpoint
# from mmdet.datasets.pipelines
import itertools
from mmcv.parallel import DataContainer as DC
import torch
from model_2d.meshrcnn_roi_head import MeshRCNNROIHead
from model_2d.voxel_head import VoxelRCNNConvUpsampleHead
from model_2d.pix3d_dataset import Pix3DDataset
import trimesh


def test_single_data(dataset, index):
    data = dataset[index]
    data['img'] = torch.stack(data['img'])
    data['img_metas'] = [data['img_metas'][0].data]
    return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    cfg_path = 'config/meshrcnn_r50_fpn_1x.py'
    cfg = Config.fromfile(cfg_path)
    m = build_detector(cfg.model)
    m.eval()
    load_checkpoint(m, '/home/hanyang/amodal_grisp/work_dirs/meshrcnn_r50_fpn_1x/epoch_12.pth')
    dataset = build_dataset(cfg.data.test)
    data = test_single_data(dataset, 104)
    with torch.no_grad():
        res = m.simple_test(**data)
    img = data['img'].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    # plt.imshow(img)
    # plt.show()

    # for i in res[0]:
    #     if i.__len__() != 0:
    #         for j in i:
    #             if len(j) != 0:
    #                 for k in j:
    #                     box, score = k[:4], k[4]
    #                     if score > 0.9:
    #                         cv2.rectangle(img, np.int0(box[:2]), np.int32(box[2:]), (0, 255, 0), 5)
    plt.imshow(img)
    plt.show()


    mesh_out = res[2][1]
    voxel = res[2][0][0]
    faces = mesh_out[0][1].cpu().numpy()
    verts = mesh_out[0][0].cpu().numpy()
    mesh = trimesh.Trimesh(faces=faces, vertices=verts)
    mesh.export('/home/hanyang/2.obj')

    # mesh_pred = res[2]['mesh_pred']
    # a = 0
    # for i in mesh_pred:
    #     verts, faces = i
    #     verts = verts.detach().cpu().numpy()
    #     faces = faces.detach().cpu().numpy()
    #     mesh = trimesh.Trimesh(faces=faces,
    #                            vertices=verts)
    #     mesh.export(f'/home/hanyang/{a}.obj')
    #     a += 1

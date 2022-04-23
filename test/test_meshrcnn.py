from mmdet.models.builder import build_detector
from mmcv.utils import Config
from mmdet.datasets.builder import build_dataloader, build_dataset
import itertools
from mmcv.parallel import DataContainer as DC
import torch
from model_2d.meshrcnn_roi_head import MeshRCNNROIHead
from model_2d.voxel_head import VoxelRCNNConvUpsampleHead
from model_2d.pix3d_dataset import Pix3DDataset
import trimesh

def test_single_data(dataset, index):
    data = dataset[index]
    new_data = dict()
    for k, v in data.items():
        if k != 'img_metas':
            if isinstance(v, DC):
                v: DC
                v = v.data
                if k == 'img':
                    new_data[k] = v.unsqueeze(0)
                else:
                    new_data[k] = [v]
            elif isinstance(v, torch.Tensor):
                new_data[k] = v.unsqueeze(0)
            else:
                new_data[k] = v
        else:
            new_data[k] = [v.data]
    return new_data




if __name__ == '__main__':
    cfg_path = 'config/meshrcnn_r50_fpn_1x.py'
    cfg = Config.fromfile(cfg_path)
    m = build_detector(cfg.model)
    m.eval()
    dataset = build_dataset(cfg.data.train)
    data_new = test_single_data(dataset, 0)

    res = m.forward_train(**data_new)


    mesh_pred = res[2]['mesh_pred']
    for i in mesh_pred:
        verts, faces = i
        verts = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(faces=faces,
                               vertices=verts)




    # dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0)
    # itertools.islice(dataloader, 15, None)
    # data = dataloader.__iter__().next()


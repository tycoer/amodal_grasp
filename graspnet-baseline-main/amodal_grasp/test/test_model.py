from mmdet.models.builder import build_detector
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmcv.utils.config import Config
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt
import numpy as np
import torch

from amodal_grasp.dataset import *
from amodal_grasp.models import *

if __name__ == '__main__':
    cfg_path = 'configs/amodal_grasp_pybullet_cfg.py'
    checkpoint_path = '/home/guest/graspnet-baseline-main/work_dirs/amodal_grasp_pybullet_cfg/epoch_4.pth'

    cfg = Config.fromfile(cfg_path)
    model = build_detector(cfg.model)
    load_checkpoint(model=model, filename=checkpoint_path)

    dataset = build_dataset(cfg.data.test)
    # cfg.data.test.pipeline = None
    # dataset_raw = build_dataset(cfg.data.test)

    # data_raw = dataset_raw[0]
    data = dataset[0]

    dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0)
    data = dataloader.__iter__().__next__()
    data['img_metas'] = data['img_metas'][0].data[0]
    scale_factor = data['img_metas'][0]['scale_factor']
    data['img_metas'][0]['scale_factor'] = np.array([scale_factor, scale_factor, scale_factor, scale_factor])
    data['img'] = data['img'][0]

    model.eval()
    with torch.no_grad():
        res = model.simple_test(rescale=False,
                                **data)
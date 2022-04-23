from mmdet.models.builder import build_detector
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmcv.utils.config import Config
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt

from dataset import *
from model_2d import *


cfg_path = 'config/amodal_grisp_r50_fpn_1x.py'
checkpoint_path = '/home/hanyang/amodal_grisp/work_dirs/nocs_r50_fpn_1x/epoch_7.pth'

cfg = Config.fromfile(cfg_path)
model = build_detector(cfg.model)
load_checkpoint(model=model, filename=checkpoint_path)


dataset = build_dataset(cfg.data.test)
cfg.data.test.pipeline = None
dataset_raw = build_dataset(cfg.data.test)

data_raw = dataset_raw[0]
data = dataset[0]

rgb = data_raw['img']
plt.imshow(rgb)
plt.show()

dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0)
data = dataloader.__iter__().__next__()
data['img_metas'] = data['img_metas'].data[0]
model.with_nocs = False
res = model.simple_test(**data)
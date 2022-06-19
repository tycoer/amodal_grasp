from mmdet.models.builder import build_detector
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmcv.utils.config import Config
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt
import torch
from dataset import *
from model_2d import *
import cv2
import numpy as np


cfg_path = 'config/amodal_grasp_only_grasp_no_mmdet_cfg.py'
checkpoint_path = '/home/hanyang/amodal_grisp/work_dirs/nocs_r50_fpn_1x/epoch_10.pth'

cfg = Config.fromfile(cfg_path)
model = build_detector(cfg.model)
load_checkpoint(model=model, filename=checkpoint_path)


dataset = build_dataset(cfg.data.test)
cfg.data.test.pipeline = None
dataset_raw = build_dataset(cfg.data.test)

data_raw = dataset_raw[0]
data = dataset[0]

# rgb = data_raw['img']
# plt.imshow(rgb)
# plt.show()

model = model.eval()
res_list = []
gt_qual_list = []
dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0)
for i, data in enumerate(dataloader):
    # data = .__iter__().__next__()
    data['img_metas'] = data['img_metas'].data[0]
    model.with_nocs = False
    with torch.no_grad():
        res = model.simple_test(**data)
    img = data['img'][-1][-1]
    img = cv2.normalize(img.numpy(), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img = img.reshape(480, 480, 1).repeat(3, -1)
    for vu in np.int32(data['img_metas'][0]['gt_uv_obj']):
        cv2.drawMarker(img, vu, (255, 0, 0), 1, 5, 5)
    plt.imshow(img)
    plt.show()
    plt.imshow(res[0])
    plt.show()
    print(res.max())
    res_list.append(res)
    if i == 20:
        break
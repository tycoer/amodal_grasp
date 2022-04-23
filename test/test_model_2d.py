from mmdet.models.builder import build_detector
from mmdet.datasets.builder import build_dataset, build_dataloader
from mmcv.utils.config import Config
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from dataset import *
from model_2d import *

def batch_argmax(tensor):
    '''
    Locate uv of max value in a tensor in dim 2 and dim 3.
    The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).

    Args:
        tensor: torch.tensor

    Returns: uv of max value with batch.

    '''
    # if tensor.shape = (16, 3, 64, 64)
    assert tensor.ndim == 4, 'The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).'
    b, c, h, w = tensor.shape
    tensor = tensor.reshape(b, c, -1)  # flatten the tensor to (16, 3, 4096)
    idx = tensor.argmax(dim=2)
    v = idx // w    # shape (16, 3)
    u = idx - v * w  # shape (16, 3)
    u, v = u.unsqueeze(-1), v.unsqueeze(-1)  # reshape u, v to (16, 3, 1) for cat
    uv = torch.cat((u, v), dim=-1)  # shape (16, 3, 2)
    return uv

example_index = 12

cfg_path = 'config/amodal_grasp_r50_fpn_1x.py'
checkpoint_path = '/home/hanyang/amodal_grisp/work_dirs/nocs_r50_fpn_1x/epoch_32.pth'

cfg = Config.fromfile(cfg_path)
model = build_detector(cfg.model)
load_checkpoint(model=model, filename=checkpoint_path)


dataset = build_dataset(cfg.data.test)
cfg.data.test.pipeline = None
dataset_raw = build_dataset(cfg.data.test)

data_raw = dataset_raw[example_index]
data = dataset[example_index]

rgb = data_raw['img']
plt.imshow(rgb)
plt.show()

# dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0)
# data = dataloader.__iter__().__next__()
# data['img_metas'] = data['img_metas'].data[0]
# data['img'] = data['img'].float()

img = data['img'].unsqueeze(0).float()
with torch.no_grad():
    pred_heatmap, pred_qual, pred_width, pred_quat = model.forward_grasp(img)
pred_qual = pred_qual.cpu().numpy()
valid_grasp_index = np.squeeze(pred_qual >= 0.5, 0)


pred_uv_for_hm = batch_argmax(pred_heatmap)
pred_uv_for_hm = pred_uv_for_hm.cpu().numpy()
pred_uv = pred_uv_for_hm / np.array([[40, 40]]) * np.array([[640, 640]])

############## visualize uv #################
# visualize gt
gt_gripper_uv = data_raw['gt_gripper_T_uv']
gt_gripper_qual = data_raw['gt_gripper_qual']
for uv, qual in zip(gt_gripper_uv, gt_gripper_qual):
    if qual == 1:
        cv2.drawMarker(rgb, np.int0(uv), (0, 0, 255), 5)

# visualize pred
for i, vaild in enumerate(valid_grasp_index):
    if vaild == True:
        cv2.drawMarker(rgb, np.int0(pred_uv[0, i]), (0, 255, 0), 5)
    else:
        cv2.drawMarker(rgb, np.int0(pred_uv[0, i]), (255, 0, 0), 5)


plt.imshow(rgb)
plt.show()


valid_width = pred_width.cpu().numpy()[:,valid_grasp_index]
valid_quat = pred_quat.cpu().numpy()[:, valid_grasp_index]



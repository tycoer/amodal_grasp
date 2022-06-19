import numpy as np
import cv2
from mmdet.models.dense_heads import rpn_head
# 用一个numpy array代表我们的图像
img = np.array([[3, 106, 107, 40, 148, 112, 254, 151],
                      [62, 173, 91, 93, 33, 111, 139, 25],
                      [99, 137, 80, 231, 101, 204, 74, 219],
                      [240, 173, 85, 14, 40, 230, 160, 152],
                      [230, 200, 177, 149, 173, 239, 103, 74],
                      [19, 50, 209, 82, 241, 103, 3, 87],
                      [252, 191, 55, 154, 171, 107, 6, 123],
                      [7, 101, 168, 85, 115, 103, 32, 11]],
                      dtype=np.uint8)
# 把宽和高缩小为原来的一半
resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)),
                            interpolation=cv2.INTER_NEAREST_EXACT)
# resized1 = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2),
#                             interpolation=cv2.INTER_MAX)
# resized = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2),
#                             interpolation=cv2.INTER_AREA)
# resized = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2),
#                             interpolation=cv2.INTER_AREA)
print(resized)
from torchvision.models.detection import maskrcnn_resnet50_fpn
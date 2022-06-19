import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
if __name__ == '__main__':
    seg_path = '/disk5/data/graspnet/scenes/scene_0186/kinect/label/0161.png'
    seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    # cv2
    start = time.time()
    for i in np.unique(seg):
        if i == 0:
            continue
        mask = seg == i
        cnts = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
        box = cv2.boundingRect(cnts[0])
    print(time.time() - start)

    # numpy
    start = time.time()
    for i in np.unique(seg):
        m = seg == i
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
    print(time.time() - start)

    # skimage
    start = time.time()
    prop = regionprops(seg)
    box = [i.bbox for i in prop]
    print(time.time() - start)

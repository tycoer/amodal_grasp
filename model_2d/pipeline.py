import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class SimplePadding:
    def __init__(self,
                 out_shape=(640, 640)):
        self.out_shape = out_shape
    def __call__(self, results):
        img = results['img']
        h, w, d = img.shape
        padding_shape = self.out_shape + (d,)
        padding = np.zeros(padding_shape, dtype='float32')
        padding[:h, :w, :] = img

        results['img'] = padding
        results['pad_shape'] = self.out_shape
        return results



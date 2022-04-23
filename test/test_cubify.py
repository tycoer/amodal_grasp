from pytorch3d.ops import cubify
import torch
if __name__ == '__main__':
    vox = torch.rand(10, 24, 24, 24)

    res = cubify(vox, 0.2)
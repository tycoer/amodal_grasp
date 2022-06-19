from model_2d.utils.decoder import LocalDecoder
from model_2d.grasp_head_for_grid import GripHead, Decoder
import torch


if __name__ == '__main__':
    p = torch.rand(32, 1, 3)
    c = torch.rand(32, 32, 40, 40)
    decoder = Decoder()
    res = decoder(p, c)
    gt_qual = torch.rand(32)
    gt_rot = torch.rand(32, 1, 4)
    gt_width = torch.rand(32)
    head = GripHead()
    res = head.forward_train(x=c,
                             gripper_T=p,
                             gt_qual=gt_qual,
                             gt_rotation=gt_rot,
                             gt_width=gt_width)

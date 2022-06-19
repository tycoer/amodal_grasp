import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from mmdet.models.builder import HEADS
from model_2d.utils.utils import normalize_coordinate, ResnetBlockFC

class Decoder(nn.Module):
    def __init__(self,
                 dim=3,
                 c_dim=32,
                 hidden_size=32,
                 out_dim=1,
                 n_blocks=5,
                 leaky=False,
                 sample_mode='bilinear',
                 padding=0.1,
                 concat_feat=False,
                 no_xyz=False):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        # tycoer
        self.fc_init = nn.Linear(256, 32)


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def forward(self, p, c):
        c = self.sample_plane_feature(p, c)
        c = c.transpose(1, 2)
        net = self.fc_p(p)
        c = self.fc_init(c)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out

@HEADS.register_module()
class GripHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_qual = Decoder(out_dim=1)
        self.decoder_rot = Decoder(out_dim=4)
        self.decoder_width = Decoder(out_dim=1)

    def decode_grasp(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        # tycoer
        qual = qual.squeeze(-1)
        width = width.squeeze(-1)
        rot = rot.squeeze(1)
        return qual, rot, width


    def forward_train(self,
                      # gt
                      gt_qual,
                      gt_rotation,
                      gt_width,
                      # input
                      x,  # shape (bz, 32, 40, 40)
                      gripper_T,  # gripper xyz 平移 shape (bz, 1, 3)
                      ):
        # grid_size = (x.shape[-1], x.shape[-2])
        # gripper_T[:, :, :2] /= grid_size
        pred_qual, pred_rotation, pred_width = self.decode_grasp(p=gripper_T, c=x)
        loss_qual, loss_rot, loss_width = self.loss_grasp(pred_qual,
                                                             pred_rotation,
                                                             pred_width,
                                                             gt_qual,
                                                             gt_rotation,
                                                             gt_width)
        return dict(loss_qual=loss_qual, loss_rot=loss_rot, loss_width=loss_width)


    def forward_test(self,
                     p,
                     c,
                     ):
        return self.decode_grasp(p, c)


    def loss_grasp(self,
             pred_qual,
             pred_rotation,
             pred_width,
             gt_qual,
             gt_rotation,
             gt_width):
        loss_qual = self._qual_loss_fn(pred_qual, gt_qual)  # ；抓取质量
        loss_rot = self._quat_loss_fn(pred_rotation, gt_rotation)  # 夹爪旋转
        loss_width = self._width_loss_fn(pred_width, gt_width)  # 夹爪开合宽度
        return loss_qual, loss_rot, loss_width


    def _qual_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="mean")

    # def _rot_loss_fn(self, pred, target):
    #     loss0 = self._quat_loss_fn(pred, target[:, 0])
    #     loss1 = self._quat_loss_fn(pred, target[:, 1])
    #     return torch.min(loss0, loss1)

    def _quat_loss_fn(self, pred, target):
        return torch.mean(1.0 - torch.abs(torch.sum(pred * target, dim=1)))

    def _width_loss_fn(self, pred, target):
        return F.mse_loss(40 * pred, 40 * target, reduction="mean")


if __name__ == '__main__':
    p = torch.rand(32, 1, 3)
    c = torch.rand(32, 32, 40, 40)
    # decoder = Decoder()
    # res = decoder(p, c)
    gt_qual = torch.rand(32)
    gt_rot = torch.rand(32, 1, 4)
    gt_width = torch.rand(32)
    head = GripHead()
    res = head.forward_train(x=c,
                             gripper_T=p,
                             gt_qual=gt_qual,
                             gt_rotation=gt_rot,
                             gt_width=gt_width)

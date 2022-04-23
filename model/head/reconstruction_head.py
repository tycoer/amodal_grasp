import torch.nn as nn
from model.head.utils.decoder import LocalDecoder
import torch.functional as F


class ReconstructionHead(nn.Module):
    def __init__(self,
                 dim=3,
                 c_dim=128,
                 hidden_size=256,
                 n_blocks=5,
                 out_dim=1,
                 leaky=False,
                 sample_mode='bilinear',
                 padding=0.1,
                 concat_feat=False,
                 no_xyz=False
                 ):
        super().__init__()
        model_cfg = dict(
                         dim=3,
                         c_dim=128,
                         hidden_size=256,
                         n_blocks=5,
                         out_dim=1,
                         leaky=False,
                         sample_mode='bilinear',
                         padding=0.1,
                         concat_feat=False,
                         no_xyz=False)

        self.decoder_tsdf = LocalDecoder(**model_cfg)

    def forward(self, occ , features):
        tsdf = self.decoder_tsdf(p_tsdf, features)
        return tsdf

    def forward_train(self,
                      p_tsdf,
                      c,
                      gt_occ):
        pred_occ = self.forward(p_tsdf, c)
        loss_occ = self.loss(pred_occ, gt_occ)
        losses = dict(loss_occ=loss_occ)
        return losses

    def forward_test(self,
                     p_tsdf,
                     c):
        return self.forward(p_tsdf, c)

    def loss(self,
             pred_occ,
             gt_occ):
        loss_occ = self._occ_loss_fn(pred_occ, gt_occ)
        return loss_occ

    def _occ_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)

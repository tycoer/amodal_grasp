import torch
import torch.nn as nn
from .nocs3d_head_utils import MirrorMSELoss, VirtualGrid
import torch.functional as F

class NOCS3DHead(torch.nn.Module):
    def __init__(self,
                 nocs_bin=32,
                 dropout=True,
                 ):
        super().__init__()
        self.nocs_bin = nocs_bin
        out_channels = nocs_bin * 3
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, out_channels)

        self.dp1 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x
        self.dp2 = nn.Dropout(p=0.5, inplace=False) if dropout else lambda x: x

        # grid
        self.vg = self.get_virtual_grid()

        # loss
        self.nocs_loss = nn.CrossEntropyLoss()

    def forward(self,
                x,
                # global_features
                ):
        # pre-point prediction
        x = F.relu(self.lin1(x))
        x = self.dp1(x)
        x = self.lin2(x)
        features = self.dp2(x)
        logits = self.lin3(features)
        return logits

    def get_virtual_grid(self):
        nocs_bins = self.nocs_bins
        device = self.device
        vg = VirtualGrid(lower_corner=(0, 0, 0), upper_corner=(1, 1, 1),
                         grid_shape=(nocs_bins,) * 3, batch_size=1,
                         device=device, int_dtype=torch.int64,
                         float_dtype=torch.float32)
        return vg


    def forward_train(self,
                      x, # backbone features
                      gt_nocs,
                      ):
        pred_logits = self.forward(x)
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], self.nocs_bins, 3))
        gt_nocs_idx = self.vg.get_points_grid_idxs(gt_nocs)
        nocs_loss = self.nocs_loss(pred_logits_bins, gt_nocs_idx)
        losses = dict(nocs_loss=nocs_loss)
        return losses


    def forward_test(self,
                     x,  # backbone features
                     ):
        pred_logits = self.forward(x)
        pred_logits_bins = pred_logits.reshape(
            (pred_logits.shape[0], self.nocs_bins, 3))
        nocs_bin_idx_pred = torch.argmax(pred_logits_bins, dim=1)
        pred_nocs = self.vg.idxs_to_points(nocs_bin_idx_pred)
        return pred_nocs

    def eval(self,
             pred_nocs,
             gt_nocs,
             ):
        nocs_err_dist = torch.norm(pred_nocs - gt_nocs, dim=-1).mean()
        return nocs_err_dist

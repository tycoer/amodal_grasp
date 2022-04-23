import torch
import torch.nn as nn
from model.head.utils.decoder import LocalDecoder
from model.head.utils.encoder import LocalVoxelEncoder
import torch.functional as F
from torch import distributions as dist
from mmdet.models.builder import HEADS


@HEADS.register_module()
class GripHead(nn.Module):
    def __init__(self,

                 ):
        super().__init__()
        decoder_cfg = dict(
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
                        )

        encoder_cfg = dict(dim=2,
                           c_dim=32,
                           plane_resolution=512,
                           grid_resolution=None,
                           plane_type='xz',
                           kernel_size=3,
                           padding=0.1)


        self.encoder = LocalVoxelEncoder(**encoder_cfg)
        self.decoder_qual = LocalDecoder(**decoder_cfg)
        self.decoder_rot = LocalDecoder(**decoder_cfg)
        self.decoder_width = LocalDecoder(**decoder_cfg)
        self.decoder_occ = LocalDecoder(**decoder_cfg)

    def forward(self,
                voxel_grid_features, # shape (bz, 2, 40, 40, 40)
                gripper_T=None,  # gripper xyz 平移 shape (bz, 1, 3)
                occ_point=None,  # shape (bz, 2048, 3)
                ):
        encoder_features = self.encoder(voxel_grid_features)
        # encoder_features is a dict
        # {'xy': (bz, 32, 40, 40)}
        # {'xz': (bz, 32, 40, 40)}
        # {'yz': (bz, 32, 40, 40)}

        if gripper_T is not None and occ_point is None:
            qual, rot, width = self.decode(gripper_T, encoder_features)
            return qual, rot, width
        elif gripper_T is not None and occ_point is not None:
            # detch encoder features
            qual, rot, width = self.decode(gripper_T, encoder_features)
            encoder_features =  {k: v.detach() for k, v in encoder_features}
            tsdf =  self.decoder_occ(occ_point, encoder_features)
            return qual, rot, width, tsdf
        elif gripper_T is None and occ_point is not None:
            tsdf =  self.decoder_occ(occ_point, encoder_features)
            return tsdf
        else:
            raise ValueError



    # def forward_occ(self,
    #                            voxel_grid_features,
    #                            occ_points):
    #     encoder_features = self.encoder(voxel_grid_features)
    #     tsdf = self.decoder_occ(occ_points, encoder_features)
    #     return tsdf
    #
    #
    # def forward_grip(self,
    #                  voxel_grid_features,
    #                  gripper_T,
    #                  ):
    #     encoder_features = self.encoder(voxel_grid_features)
    #     qual, rot, width = self.decode(gripper_T, encoder_features)
    #     return qual, rot, width
    #
    # def forward_grip_and_occ(self,
    #                                     voxel_grid_features,
    #                                     gripper_T,
    #                                     ):



    def decode_grip(self, p, c, **kwargs):
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
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r


    def forward_train(self,
                      # gt
                      gt_label,
                      gt_rotation,
                      gt_width,
                      gt_occ,

                      # input
                      voxel_grid_features,  # shape (bz, 2, 40, 40, 40)
                      gripper_T=None,  # gripper xyz 平移 shape (bz, 1, 3)
                      occ_points=None,  # shape (bz, 2048, 3)



                      ):

        if gripper_T is not None and occ_points is None:
            pred_label, pred_rotation, pred_width = self.forward(voxel_grid_features, gripper_T, occ_points)
            loss_qual, loss_rot, loss_width = self.loss(pred_label,
                                                        pred_rotation,
                                                        pred_width,
                                                        gt_label,
                                                        gt_rotation,
                                                        gt_width
                                                        )
            losses = dict(loss_qual=loss_qual,
                          loss_rot=loss_rot,
                          loss_width=loss_width,
                          )
            return losses


        elif gripper_T is not None and occ_points is not None:
            pred_label, pred_rotation, pred_width, pred_tsdf = self.forward(voxel_grid_features, gripper_T, occ_points)

            loss_qual, loss_rot, loss_width = self.loss_grip(pred_label,
                                                        pred_rotation,
                                                        pred_width,
                                                        gt_label,
                                                        gt_rotation,
                                                        gt_width
                                                        )
            loss_occ = self.loss_occ(pred_tsdf, gt_occ)
            losses = dict(loss_qual=loss_qual,
                          loss_rot=loss_rot,
                          loss_width=loss_width,
                          loss_occ = loss_occ
                          )
            return losses


        elif gripper_T is None and occ_points is not None:
            pred_tsdf = self.forward(voxel_grid_features, gripper_T, occ_points)
            loss_occ = self.loss_occ(pred_tsdf, gt_occ)
            losses = dict(loss_occ)
            return losses
        else:
            raise ValueError


    def forward_test(self,
                     p,
                     c,
                     ):
        return self.forward(p, c)


    def loss_grip(self,
             pred_label,
             pred_rotation,
             pred_width,
             gt_label,
             gt_rotation,
             gt_width):
        loss_qual = self._qual_loss_fn(pred_label, gt_label)  # ；抓取质量
        loss_rot = self._rot_loss_fn(pred_rotation, gt_rotation)  # 夹爪旋转 + 平移
        loss_width = self._width_loss_fn(pred_width, gt_width)  # 夹爪开合宽度
        return loss_qual, loss_rot, loss_width


    def loss_occ(self,
                pred_occ,
                gt_occ):
        loss_occ = self._occ_loss_fn(pred_occ, gt_occ)
        return loss_occ


    def _qual_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none")

    def _rot_loss_fn(self, pred, target):
        loss0 = self._quat_loss_fn(pred, target[:, 0])
        loss1 = self._quat_loss_fn(pred, target[:, 1])
        return torch.min(loss0, loss1)

    def _quat_loss_fn(self, pred, target):
        return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

    def _width_loss_fn(pred, target):
        return F.mse_loss(40 * pred, 40 * target, reduction="none")


    def _occ_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)

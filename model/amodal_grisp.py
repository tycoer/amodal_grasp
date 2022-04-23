import torch.nn as nn
import torch
from mmdet.models.builder import build_backbone, build_head

class AmodalGrisp(nn.Module):
    def __init__(self,
                 backbone,
                 instance_seg_head,
                 nocs_head=None,
                 reconstruction_head=None,
                 grip_head=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.instance_seg_head = build_head(instance_seg_head)

        # nocs_head
        if nocs_head is not None:
            self.nocs_head = build_head(nocs_head)
            self.nocs_head = True
        else:
            self.nocs_head = nocs_head
            self.with_nocs_head = False

        # reconstruction_head
        if reconstruction_head is not None:
            self.reconstuction_head = build_head(reconstruction_head)
            self.with_reconstruction_head = True
        else:
            self.reconstuction_head = reconstruction_head
            self.with_reconstruction_head = False

        # grip_head
        if grip_head is not None:
            self.grip_head = build_head(grip_head)
            self.with_grip_head = True
        else:
            self.grip_head = grip_head
            self.with_grip_head = False

    def forward_train_nocs(self):
        nocs_loss = self.nocs_head.forward_train()
        return nocs_loss


    def forward_train_instance_seg(self):
        seg_loss = self.instance_seg_head.forward_train()
        return seg_loss

    def forward_train_grip(self):
        grip_loss = self.grip_head.forward_train()
        return grip_loss

    def forward_train_reconstruction(self):
        reconstruction_loss = self.reconstuction_head.forward_train()
        return reconstruction_loss

    def forward_test_nocs(self):
        nocs = self.nocs_head.forward_test()
        return nocs

    def forward_test_instance_seg(self):
        pass

    def forward_test_grip(self):
        qual, rotation, width = self.grip_head.forward_test()
        return qual, rotation, width

    def forward_test_reconstruction(self):
        tsdf = self.reconstuction_head.forward_test()
        return tsdf

    def forward_train(self):
        features = self.backbone()
        losses = {}
        seg_loss = self.forward_train_instance_seg()
        losses.update(seg_loss)
        if self.with_nocs_head:
            nocs_loss = self.nocs_head.forward_train()
            losses.update(nocs_loss)
        if self.with_reconstruction_head:
            reconstruction_loss = self.reconstuction_head.forward_train()
            losses.update(reconstruction_loss)
        if self.with_grip_head:
            grip_loss = self.grip_head.forward_train()
            losses.update(grip_loss)
        return losses

    def forward_test(self):
        # nocs
        pred_nocs = self.nocs_head.forward_test()
        # grip
        # grip quality, gripper H(4x4), gripper_width
        pred_qual, pred_rotation, pred_width = self.grip_head.forward_test()

        pred_tsdf = self.reconstuction_head.forward_test()

        results = dict(pred_nocs=pred_nocs,
                       pred_qual=pred_qual,
                       pred_rotation=pred_rotation,
                       pred_width=pred_width,
                       pred_tsdf=pred_tsdf,




                       )
        return results


    def loss(self):
        pass


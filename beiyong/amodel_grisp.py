import torch
from .nocs import NOCS
from mmdet.models.builder import build_head

class AmodelGrisp(NOCS):
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 nocs_roi_extractor=None,
                 nocs_head=None,
                 shared_head=None,
                 pretrained=None,
                 grisp_head=None,
                 reconstruction_head=None,
                 ):
        super().__init__(
                 backbone=backbone,
                 rpn_head=rpn_head,
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 neck=neck,
                 nocs_roi_extractor=nocs_roi_extractor,
                 nocs_head=nocs_head,
                 shared_head=shared_head,
                 pretrained=pretrained)


        if grisp_head is not None:
            self.with_grisp_head = True
            self.grisp_head = build_head(grisp_head)
        else:
            self.with_grisp_head = False
            self.grisp_head = grisp_head

        if reconstruction_head is not None:
            self.with_reconstruction_head = True
            self.reconstruction_head = build_head(reconstruction_head)
        else:
            self.with_reconstruction_head = False
            self.reconstruction_head = reconstruction_head

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      depth=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_coords=None,
                      proposals=None,
                      scales=None,
                      ):
        super().forward_train(img=img,
                              img_meta=img_meta,
                              gt_bboxes=gt_bboxes,
                              gt_labels=gt_labels,
                              depth=depth,
                              )

from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.builder import HEADS, build_roi_extractor, build_head
import torch

@HEADS.register_module()
class AmodalGraspROIHead(StandardRoIHead):
    def __init__(self,
                 nocs_head=None,
                 nocs_roi_extractor=None,

                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AmodalGraspROIHead, self).__init__(bbox_head=bbox_head,
                                                 bbox_roi_extractor=bbox_roi_extractor,
                                                 mask_head=mask_head,
                                                 mask_roi_extractor=mask_roi_extractor,
                                                 shared_head=shared_head,
                                                 train_cfg=train_cfg,
                                                 init_cfg=init_cfg,
                                                 pretrained=pretrained,
                                                 test_cfg=test_cfg)

        self.init_nocs_head(nocs_roi_extractor, nocs_head)

    def init_nocs_head(self, nocs_roi_extractor, nocs_head):
        self.nocs_head = nocs_head
        self.nocs_roi_extractor = nocs_roi_extractor
        self.with_nocs = False
        if self.nocs_head is not None:
            self.nocs_head = build_head(nocs_head)
            self.with_nocs = True
        if self.nocs_roi_extractor is not None:
            self.nocs_roi_extractor = build_roi_extractor(nocs_roi_extractor)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs
                      ):
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses
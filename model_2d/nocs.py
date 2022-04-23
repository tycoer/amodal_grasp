import torch

from mmdet.models.detectors import TwoStageDetector
from mmdet.models import builder
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from .coord import align

@DETECTORS.register_module()
class NOCS(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 # roi_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 nocs_roi_extractor=None,
                 nocs_head=None,
                 init_cfg=None):
        super(NOCS, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)


        self.bbox_roi_extractor = builder.build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = builder.build_head(bbox_head)
        self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
        self.mask_head = builder.build_head(mask_head)

        self.nocs_roi_extractor = builder.build_roi_extractor(nocs_roi_extractor)
        self.nocs_roi_extractor.init_weights()
        self.nocs_head = builder.build_head(nocs_head)
        self.nocs_head.init_weights()

    @property
    def with_nocs(self):
        return hasattr(self, 'nocs_head') and self.nocs_head is not None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      depth=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_coords=None,
                      proposals=None,
                      scales=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            scale_factor = img_metas[0]['scale_factor']
            cls_score, bbox_pred = self.bbox_head(bbox_feats, rois=rois, scale_factor=scale_factor)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.size(0) > 0:
                mask_pred = self.mask_head(mask_feats, rois=pos_rois, scale_factor=scale_factor)

                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        if self.with_nocs:
            assert img_metas[0]['domain'] == img_metas[1]['domain']
            domain = img_metas[0]['domain']
            if domain in ['camera', 'real']:
                if not self.share_roi_extractor:
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    nocs_feats = self.nocs_roi_extractor(
                        x[:self.nocs_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        nocs_feats = self.shared_head(nocs_feats)
                else:
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    nocs_feats = bbox_feats[pos_inds]
                if nocs_feats.size(0) > 0:
                    nocs_pred = self.nocs_head(nocs_feats, rois=pos_rois, scale_factor=scale_factor)

                    nocs_targets = self.nocs_head.get_target(
                        sampling_results, gt_coords, self.train_cfg.rcnn)
                    pos_labels = torch.cat(
                        [res.pos_gt_labels for res in sampling_results])
                    loss_nocs = self.nocs_head.loss(nocs_pred, nocs_targets,
                                                    pos_labels, mask_targets)
                    losses.update(loss_nocs)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        return self.simple_test(img, img_metas[0], **kwargs)

    def simple_test(self, img, img_meta, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert 'depth' in kwargs
        depth = kwargs.get('depth')[0]

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        elif not self.with_nocs:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale, encode=False)
            nocs_results = self.simple_test_nocs(
                x, img_meta, det_bboxes, det_labels, rescale=rescale
            )
            RT_results, scale_results, _, _ = align(bbox_results, segm_results, nocs_results, depth, det_labels, img_meta)
            return RT_results, scale_results, bbox_results, segm_results, nocs_results, img_meta

    def simple_test_nocs(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        window = img_meta[0]['window'] if 'window' in img_meta[0] else (0, 0, 0, 0)
        if det_bboxes.shape[0] == 0:
            nocs_result = [[] for _ in range(self.nocs_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            nocs_rois = bbox2roi([_bboxes])
            nocs_feats = self.nocs_roi_extractor(
                x[:len(self.nocs_roi_extractor.featmap_strides)], nocs_rois)
            if self.with_shared_head:
                nocs_feats = self.shared_head(nocs_feats)
            nocs_pred = self.nocs_head(nocs_feats, rois=nocs_rois, scale_factor=scale_factor)
            nocs_result = self.nocs_head.get_nocs(
                nocs_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, window, rescale)
        return nocs_result

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

from mmdet.models import MaskRCNN
from mmdet.models.builder import build_head, DETECTORS
@DETECTORS.register_module()
class AmodalGrasp(MaskRCNN):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 grasp_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.grasp_head = grasp_head
        if self.grasp_head is not None:
            self.with_grasp = True
            self.grasp_head = build_head(grasp_head)
        else:
            self.with_grasp = False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x_backbone = self.backbone(img)
        if self.with_neck:
            x = self.neck(x_backbone)

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
            proposal_list = proposals # [(1000, 5), (1000, 5)]

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)


        if self.with_grasp:
            gt_heatmap, gt_qual, gt_width, gt_quat = kwargs['gt_heatmaps'], \
                                                     kwargs['gt_gripper_qual'], \
                                                     kwargs['gt_gripper_width'], \
                                                     kwargs['gt_gripper_quat']
            pred_heatmap, pred_qual, pred_width, pred_quat = self.grasp_head(x_backbone[-1])
            grasp_losses = self.grasp_head.loss(pred_heatmap,
                                   pred_qual,
                                   pred_width,
                                   pred_quat,
                                   gt_heatmap,
                                   gt_qual,
                                   gt_width,
                                   gt_quat)
            losses.update(grasp_losses)

        return losses

    def forward_grasp(self, img):
        x_backbone = self.backbone(img)
        pred_heatmap, pred_qual, pred_width, pred_quat = self.grasp_head(x_backbone[-1])
        return pred_heatmap, pred_qual, pred_width, pred_quat


        




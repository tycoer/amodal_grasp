from mmdet.models.builder import DETECTORS, build_backbone, build_head
from mmcv.runner import BaseModule
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class AmodalGraspOnlyGrasp(BaseDetector):
    def __init__(self,
                 backbone,
                 grasp_head,
                 init_cfg=None,
                 **kwargs):
        super(AmodalGraspOnlyGrasp, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.grasp_head = build_head(grasp_head)

    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        feats = self.backbone(img)
        # feat = feats[1]

        grasp_losses = self.grasp_head.forward_train(
            x=feats,
            pos=kwargs['gt_grasp_vu_on_obj'],
            gt_qual=kwargs['gt_grasp_qual'],
            gt_quat=kwargs['gt_grasp_quat'],
            gt_width=kwargs['gt_grasp_width']
            )
        return grasp_losses



    def aug_test(self, imgs, img_metas, **kwargs):
        pass
    def extract_feat(self, imgs):
        pass
    def simple_test(self, img, img_metas, **kwargs):
        pass
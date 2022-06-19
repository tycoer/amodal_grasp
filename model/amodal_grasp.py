from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import MaskRCNN

@DETECTORS.register_module()
class AmodalGrasp(MaskRCNN):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(AmodalGrasp, self).__init__(backbone=backbone,
                                          rpn_head=rpn_head,)




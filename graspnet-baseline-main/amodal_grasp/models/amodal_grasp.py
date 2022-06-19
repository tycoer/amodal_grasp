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
                                          rpn_head=rpn_head,
                                          roi_head=roi_head,
                                          train_cfg=train_cfg,
                                          test_cfg=test_cfg,
                                          neck=neck,
                                          pretrained=pretrained,
                                          init_cfg=init_cfg
                                          )
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale, **kwargs)

    # def forward_test(self, imgs, img_metas, **kwargs):
    #     """
    #     Args:
    #         imgs (List[Tensor]): the outer list indicates test-time
    #             augmentations and inner Tensor should have a shape NxCxHxW,
    #             which contains all images in the batch.
    #         img_metas (List[List[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch.
    #     """
    #     for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
    #         if not isinstance(var, list):
    #             raise TypeError(f'{name} must be a list, but got {type(var)}')
    #
    #     num_augs = len(imgs)
    #     if num_augs != len(img_metas):
    #         raise ValueError(f'num of augmentations ({len(imgs)}) '
    #                          f'!= num of image meta ({len(img_metas)})')
    #
    #     # NOTE the batched image size information may be useful, e.g.
    #     # in DETR, this is needed for the construction of masks, which is
    #     # then used for the transformer_head.
    #     img_metas_dict = {}
    #     for img, img_meta in zip(imgs, img_metas):
    #         batch_size = len(img_meta)
    #         for img_id in range(batch_size):
    #             img_meta.update(dict(batch_input_shape = tuple(img.size()[-2:])))
    #             img_meta={img_id : img_meta}
    #     img_metas = [img_metas]
    #     if num_augs == 1:
    #         # proposals (List[List[Tensor]]): the outer list indicates
    #         # test-time augs (multiscale, flip, etc.) and the inner list
    #         # indicates images in a batch.
    #         # The Tensor should have a shape Px4, where P is the number of
    #         # proposals.
    #         if 'proposals' in kwargs:
    #             kwargs['proposals'] = kwargs['proposals'][0]
    #         return self.simple_test(imgs[0], img_metas[0], **kwargs)
    #     else:
    #         assert imgs[0].size(0) == 1, 'aug test does not support ' \
    #                                      'inference with batch size ' \
    #                                      f'{imgs[0].size(0)}'
    #         # TODO: support test augmentation for predefined proposals
    #         assert 'proposals' not in kwargs
    #         return self.aug_test(imgs, img_metas, **kwargs)

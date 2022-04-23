from mmdet.models.builder import build_head
import torch
from mmdet.models.detectors import TwoStageDetector
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from .coord import align

@DETECTORS.register_module()
class Amodalgrasp(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 init_cfg=None,
                 grasp_head=None,
                 ):
        super().__init__(
                 backbone=backbone,
                 rpn_head=rpn_head,
                 roi_head=roi_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 neck=neck,
                 init_cfg=init_cfg,
        )

        if grasp_head is not None:
            self.with_grasp_head = True
            self.grasp_head = build_head(grasp_head)
        else:
            self.with_grasp_head = False
            self.grasp_head = grasp_head
        # self.with_grasp_head = False

        # if reconstruction_head is not None:
        #     self.with_reconstruction_head = True
        #     self.reconstruction_head = build_head(reconstruction_head)
        # else:
        #     self.with_reconstruction_head = False
        #     self.reconstruction_head = reconstruction_head

        self.init_bridge_layers()

    def init_bridge_layers(self):
        self.conv_bridge = torch.nn.Conv2d(1024, 40, (1, 1))


    def forward_bridge(self, x):
        bz = x.shape[0]
        x = self.conv_bridge(x)
        x = x.reshape(bz, 1, 40, 40, 40)
        return x


    def forward_train(self,
                      img,
                      img_metass,
                      gt_bboxes,
                      gt_labels,
                      gt_coords, # tycoer
                      depth=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      #grasp_head
                      voxel_grid=None,
                      graspper_T=None,
                      occ_points=None,
                      gt_width=None ,
                      gt_qual=None,
                      gt_rotations=None,
                      gt_occ=None,

                      **kwargs
                      ):

        x, x_backbone = self.extract_feat2(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metass,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metass, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_coords, # tycoer
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # gt_labels = [i.long() for i in  gt_labels]

        if self.with_grasp_head:
            x = self.forward_bridge(x_backbone[2]) # x shape (bz, 1, 40, 40, 40)
            voxel_grid = voxel_grid.unsqueeze(1) # voxel_grid shape (bz, 1, 40, 40, 40)
            voxel_grid_features = torch.cat((x, voxel_grid), dim=1) # x shape (bz, 2, 40, 40, 40)
            loss_grasp = self.grasp_head.forward_train(voxel_grid_features=voxel_grid_features,
                                                     graspper_T=graspper_T,
                                                     occ_points=occ_points,
                                                     gt_width=gt_width,
                                                     gt_qual=gt_qual,
                                                     gt_rotation=gt_rotations,
                                                     gt_occ=gt_occ,
                                                     )
            losses.update(loss_grasp)
        return losses


    def extract_feat2(self, img):
        """Directly extract features from the backbone+neck."""
        x_bacbone = self.backbone(img)
        if self.with_neck:
            x = self.neck(x_bacbone)
        return x, x_bacbone

    def simple_test_mrcnn(self, img, img_metass, proposals=None, rescale=False):
        super().simple_test(img, img_metass, proposals, rescale)

    def simple_test_nocs(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        window = img_metas[0]['window'] if 'window' in img_metas[0] else (0, 0, 0, 0)
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



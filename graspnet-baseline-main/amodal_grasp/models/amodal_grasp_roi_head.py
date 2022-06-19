from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.builder import HEADS, build_roi_extractor, build_head
import torch
from mmdet.core import bbox2roi, bbox2result
from mmdet.models.detectors import MaskRCNN
from .utils.nocs_utils import align
import numpy as np
import warnings
# from pytorch3d.ops import cubify
# from .utils.voxel import voxel_rcnn_loss
import matplotlib.pyplot as plt

@HEADS.register_module()
class AmodalGraspROIHead(StandardRoIHead):
    def __init__(self,
                 nocs_head=None,
                 nocs_roi_extractor=None,
                 grasp_head=None,
                 mesh_head=None,
                 mesh_roi_extractor=None,
                 voxel_head=None,
                 voxel_roi_extractor=None,
                 z_head=None,
                 z_roi_extractor=None,

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
        self.init_grasp_head(grasp_head)

        if voxel_head is not None:
            self.with_voxel = True
            self.voxel_head = build_head(voxel_head)
            self.voxel_roi_extractor = build_roi_extractor(voxel_roi_extractor)
        else:
            self.with_voxel = False


        if mesh_head is not None:
            self.with_mesh = True
            self.mesh_head = build_head(mesh_head)
            self.mesh_roi_extractor = build_roi_extractor(mesh_roi_extractor)
        else:
            self.with_mesh = False


        if z_head is not None:
            self.with_z = True
            self.z_head = build_head(z_head)
            self.z_roi_extractor = build_roi_extractor(z_roi_extractor)
        else:
            self.with_z = False


    def init_nocs_head(self, nocs_roi_extractor, nocs_head):
        self.nocs_head = nocs_head
        self.nocs_roi_extractor = nocs_roi_extractor
        self.with_nocs = False
        if self.nocs_head is not None:
            self.nocs_head = build_head(nocs_head)
            self.with_nocs = True
        if self.nocs_roi_extractor is not None:
            self.nocs_roi_extractor = build_roi_extractor(nocs_roi_extractor)

    def init_grasp_head(self, grasp_head):
        self.grasp_head = grasp_head
        # self.grasp_roi_extractor = grasp_roi_extractor
        self.with_grasp = False
        if self.grasp_head is not None:
            self.grasp_head = build_head(grasp_head)
            self.with_grasp = True
        # if self.grasp_roi_extractor is not None:
        #     self.grasp_roi_extractor = build_roi_extractor(grasp_roi_extractor)

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

        if self.with_nocs:
            nocs_results = self._nocs_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    mask_results['mask_targets'],
                                                    kwargs['gt_nocs'])
            losses.update(nocs_results['loss_nocs'])

        if self.with_grasp:
            with torch.no_grad():
                nocs_pred = torch.cat(nocs_results['nocs_pred'], dim=1)
            grasp_results = self._grasp_forward_train(x=nocs_pred,
                                                      sampling_results=sampling_results,
                                                      gt_grasps=kwargs['gt_grasps'])
            losses.update(grasp_results['loss_grasp'])

        if self.with_voxel:
            shape_results = self._shape_forward_train(x=x,
                                                      sampling_results=sampling_results,
                                                      bbox_feats=bbox_results['bbox_feats'],
                                                      gt_voxels=kwargs['gt_voxels'],
                                                      gt_meshes=kwargs['gt_meshes'],
                                                      Ks=kwargs['Ks'],
                                                      img_metas=img_metas
                                                      )
            losses.update(shape_results)

        if self.with_z:
            z_results = self._z_forward_train(x,
                                              sampling_results,
                                              bbox_results['bbox_feats'],
                                              kwargs['gt_zs'],
                                              gt_labels
                                              )
            losses.update(z_results['loss_z'])
        return losses

    def _nocs_forward_train(self,
                            # input
                            x,
                            sampling_results,
                            bbox_feats,
                            mask_targets,
                            # gt
                            gt_coords,
                            ):
        if not self.share_roi_extractor:
            pos_rois = bbox2roi(
                [res.pos_bboxes for res in sampling_results])
            nocs_results = self._nocs_forward(x, pos_rois)
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
            nocs_results = self._nocs_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        nocs_targets = self.nocs_head.get_target(
            sampling_results, gt_coords, self.train_cfg)
        pos_labels = torch.cat(
            [res.pos_gt_labels for res in sampling_results])
        loss_nocs = self.nocs_head.loss(nocs_results['nocs_pred'], nocs_targets,
                                        pos_labels, mask_targets)
        nocs_results.update(loss_nocs=loss_nocs, nocs_targets=nocs_targets)
        return nocs_results


    def _grasp_forward_train(self, x, gt_grasps, sampling_results):
        qual_pred, quat_pred, width_pred, depth_pred = self.grasp_head(x)
        grasp_targets = self.grasp_head.get_target(sampling_results=sampling_results,
                                                   gt_grasps=gt_grasps,
                                                   rcnn_train_cfg=self.train_cfg)

        loss_grasp = self.grasp_head.loss(qual_pred=qual_pred,
                                          quat_pred=quat_pred,
                                          width_pred=width_pred,
                                          depth_pred=depth_pred,
                                          grasp_targets=grasp_targets)
        return dict(loss_grasp=loss_grasp)

    # def simple_test_nocs(self,
    #                      x,
    #                      img_meta,
    #                      det_bboxes,
    #                      det_labels,
    #                      rescale=False):
    #     # image shape of the first image in the batch (only one)
    #     ori_shape = img_meta[0]['ori_shape']
    #     scale_factor = img_meta[0]['scale_factor']
    #     if len(det_bboxes) == 0:
    #         nocs_result = [[] for _ in range(self.nocs_head.num_classes - 1)]
    #     else:
    #         # if det_bboxes is rescaled to the original image size, we need to
    #         # rescale it back to the testing scale to obtain RoIs.
    #         _bboxes = (
    #             det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
    #         nocs_rois = bbox2roi(_bboxes)
    #         nocs_feats = self.nocs_roi_extractor(
    #             x[:len(self.nocs_roi_extractor.featmap_strides)], nocs_rois)
    #         if self.with_shared_head:
    #             nocs_feats = self.shared_head(nocs_feats)
    #         nocs_pred = self.nocs_head(nocs_feats, rois=nocs_rois, scale_factor=scale_factor)
    #         nocs_result = self.nocs_head.get_nocs(
    #             nocs_pred, _bboxes[0], det_labels[0], self.test_cfg, ori_shape,
    #             scale_factor, rescale)
    #     return nocs_result, nocs_pred


    def simple_test_nocs(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            nocs_results = [[[] for _ in range(self.nocs_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            # if rescale:
            #     scale_factors = [
            #         torch.from_numpy(scale_factor).to(det_bboxes[0].device)
            #         for scale_factor in scale_factors
            #     ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                torch.from_numpy(scale_factors[i]).to(det_bboxes[i].device) if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            nocs_rois = bbox2roi(_bboxes)
            nocs_pred = self._nocs_forward(x, nocs_rois)['nocs_pred']

            nocs_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    nocs_results.append(
                        [[] for _ in range(self.nocs_head.num_classes)])
                else:
                    nocs_result = self.nocs_head.get_nocs(
                        nocs_pred,
                        _bboxes[i],
                        det_labels[i],
                        self.test_cfg,
                        ori_shapes[i],
                        scale_factors[i],
                        rescale)
                    nocs_results.append(nocs_result)
        return nocs_results, nocs_pred


    def _nocs_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """nocs head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            nocs_feats = self.nocs_roi_extractor(
                x[:self.nocs_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                nocs_feats = self.shared_head(nocs_feats)
        else:
            assert bbox_feats is not None
            nocs_feats = bbox_feats[pos_inds]

        nocs_pred = self.nocs_head(nocs_feats)
        nocs_results = dict(nocs_pred=nocs_pred, nocs_feats=nocs_feats)
        return nocs_results




    def simple_test_grasp(self,
                          nocs_pred):
        qual_pred, quat_pred, width_pred, depth_pred = self.grasp_head(nocs_pred)
        grasp_result = self.grasp_head.get_grasp(qual_pred, quat_pred, width_pred, depth_pred)
        return grasp_result



    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):

        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if self.with_voxel:
            shape_results = self.simple_test_shape(x, img_metas, det_bboxes, det_labels, rescale=rescale)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            if self.with_nocs:
                assert 'depth' in img_metas[0]
                depth = img_metas[0]['depth']
                nocs_results, nocs_pred = self.simple_test_nocs(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale
                )
                RT_results, scale_results, _, _ = align(bbox_results[0], segm_results[0], nocs_results[0], depth, det_labels[0], img_metas)
                if self.with_grasp:
                    nocs_pred = torch.cat(nocs_pred, dim=1)
                    grasp_pred = self.grasp_head(nocs_pred)
                    grasp_results = self.grasp_head.get_grasp(grasp_pred, det_bboxes)
                    return RT_results, scale_results, bbox_results, segm_results, nocs_results, img_metas, grasp_results
                else:
                    return RT_results, scale_results, bbox_results, segm_results, nocs_results, img_metas
            else:
                return list(zip(bbox_results, segm_results))

    # def simple_test(self,
    #                 x,
    #                 proposal_list,
    #                 img_metas,
    #                 proposals=None,
    #                 rescale=False,
    #                 **kwargs):
    #     """Test without augmentation.
    #
    #     Args:
    #         x (tuple[Tensor]): Features from upstream network. Each
    #             has shape (batch_size, c, h, w).
    #         proposal_list (list(Tensor)): Proposals from rpn head.
    #             Each has shape (num_proposals, 5), last dimension
    #             5 represent (x1, y1, x2, y2, score).
    #         img_metas (list[dict]): Meta information of images.
    #         rescale (bool): Whether to rescale the results to
    #             the original image. Default: True.
    #
    #     Returns:
    #         list[list[np.ndarray]] or list[tuple]: When no mask branch,
    #         it is bbox results of each image and classes with type
    #         `list[list[np.ndarray]]`. The outer list
    #         corresponds to each image. The inner list
    #         corresponds to each class. When the model has mask branch,
    #         it contains bbox results and mask results.
    #         The outer list corresponds to each image, and first element
    #         of tuple is bbox results, second element is mask results.
    #     """
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #
    #     det_bboxes, det_labels = self.simple_test_bboxes(
    #         x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
    #
    #     bbox_results = [
    #         bbox2result(det_bboxes[i], det_labels[i],
    #                     self.bbox_head.num_classes)
    #         for i in range(len(det_bboxes))
    #     ]
    #
    #     if not self.with_mask:
    #         return bbox_results
    #     else:
    #         segm_results = self.simple_test_mask(
    #             x, img_metas, det_bboxes, det_labels, rescale=rescale)
    #         return list(zip(bbox_results, segm_results))



    def _shape_forward_train(self,
                            x,
                            sampling_results,
                            bbox_feats,
                            gt_voxels,
                            Ks,
                            gt_meshes,
                            **kwargs
                            ):

        results = {}
        if not self.share_roi_extractor:
            pos_rois = bbox2roi(
                [res.pos_bboxes for res in sampling_results])
            voxel_feats = self.voxel_roi_extractor(x[:self.voxel_roi_extractor.num_inputs], pos_rois)
            if self.with_mesh:
                mesh_feats = self.mesh_roi_extractor(x[:self.mesh_roi_extractor.num_inputs], pos_rois)

            if self.with_shared_head:
                voxel_feats = self.shared_head(voxel_feats)
                if self.with_mesh:
                    mesh_feats = self.shared_head(mesh_feats)

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
            voxel_feats = bbox_feats[pos_inds]
            if self.with_mesh:
                mesh_feats = bbox_feats[pos_inds]

        if voxel_feats.size(0) > 0:
            voxel_pred = self.voxel_head(voxel_feats)
            voxel_targets = self.voxel_head.get_target(sampling_results=sampling_results,
                                                       gt_voxels=gt_voxels,
                                                       Ks=Ks,
                                                       **kwargs)
            loss_voxel, voxel_accuracy = self.voxel_head.loss(voxel_pred[:, 0], voxel_targets)
            results['loss_voxel'] = loss_voxel
            # results['voxel_accuracy'] = voxel_accuracy
            # results['voxel_accuracy'] = voxel_accuracy

        if self.with_mesh:
            with torch.no_grad():
                vox_in = voxel_pred.sigmoid().squeeze(1)  # (N, V, V, V)
                init_mesh = cubify(vox_in, self.voxel_head.cubify_thresh)  # 1

            mesh_pred = self.mesh_head(mesh_feats, init_mesh)

            if not mesh_pred[0].isempty():
                mesh_targets = self.mesh_head.get_target(sampling_results=sampling_results, gt_meshes=gt_meshes, Ks=Ks)
                loss_chamfer, loss_normals, loss_edge = self.mesh_head.loss(mesh_pred, mesh_targets)


            else:
                loss_chamfer = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0
                loss_normals = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0
                loss_edge = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0

            results['loss_chamfer'] = loss_chamfer
            results['loss_normals'] = loss_normals
            results['loss_edge'] = loss_edge

        return results


    def _shape_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """voxel head forward function used in both training and testing."""

        shape_results = {}
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            voxel_feats = self.voxel_roi_extractor(
                x[:self.voxel_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                voxel_feats = self.shared_head(voxel_feats)
        else:
            assert bbox_feats is not None
            voxel_feats = bbox_feats[pos_inds]

        voxel_pred = self.voxel_head(voxel_feats).sigmoid()

        shape_results['voxel_pred'] = voxel_pred

        if self.with_mesh:
            if rois is not None:
                mesh_feats = self.voxel_roi_extractor(
                    x[:self.voxel_roi_extractor.num_inputs], rois)
                if self.with_shared_head:
                    mesh_feats = self.shared_head(voxel_feats)
            else:
                assert bbox_feats is not None

            with torch.no_grad():
                vox_in = voxel_pred.squeeze(1)  # (N, V, V, V)
                init_mesh = cubify(vox_in, self.voxel_head.cubify_thresh)  # 1

            mesh_pred = self.mesh_head(mesh_feats, init_mesh)
        shape_results['mesh_pred'] = mesh_pred
        return shape_results


    def simple_test_shape(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        _bboxes = [
            det_bboxes[i][:, :4] *
            torch.from_numpy(scale_factors[i]).to(det_bboxes[i].device) if rescale else det_bboxes[i][:, :4]
            for i in range(len(det_bboxes))
        ]
        shape_rois = bbox2roi(_bboxes)
        shape_results = self._shape_forward(x, shape_rois)

        mesh_pred = shape_results['mesh_pred'][0]

        import trimesh
        for i, (f, v) in enumerate(zip(mesh_pred.faces_list(), mesh_pred.verts_list())):
            if len(f) != 0:
                mesh = trimesh.Trimesh(faces=f,
                                       vertices=v)
                mesh.export(f'/home/guest/{i}.obj')



        return shape_results

        '''
        for i, vox in enumerate(shape_results['voxel_pred'][2][0].cpu().numpy()):
            plt.imshow(vox)
            plt.show()
        
        '''

    '''
        def _z_forward_train(self,
                         # input
                         x,
                         sampling_results,
                         bbox_feats,
                         # gt
                         gt_zs,
                         gt_labels
                         ):

        if not self.share_roi_extractor:
            pos_rois = bbox2roi(
                [res.pos_bboxes for res in sampling_results])
            z_feats = self.z_roi_extractor(
                x[:self.z_roi_extractor.num_inputs], pos_rois)
            if self.with_shared_head:
                z_feats = self.shared_head(z_feats)
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
            z_feats = bbox_feats[pos_inds]

        if z_feats.size(0) > 0:
            z_pred = self.z_head(z_feats)
            loss_z = self.z_head.loss(sampling_results=sampling_results,
                                      gt_zs=gt_zs,
                                      gt_labels=gt_labels,
                                      z_pred=z_pred,)
        return dict(loss_z = dict(loss_z = loss_z))
    '''

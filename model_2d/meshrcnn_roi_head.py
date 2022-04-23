from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.builder import build_head, HEADS, build_roi_extractor
import torch
from mmdet.core import bbox2roi, bbox2result
import numpy as np
import warnings

@HEADS.register_module()
class MeshRCNNROIHead(StandardRoIHead):
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,

                 # tycoer
                 nocs_head=None,
                 nocs_roi_extractor=None,

                 voxel_head=None,
                 voxel_roi_extractor=None,

                 mesh_head = None,
                 mesh_roi_extractor=None,

                 z_head = None,
                 z_roi_extractor=None,
                 ):
        super().__init__(
                 bbox_roi_extractor=bbox_roi_extractor,
                 bbox_head=bbox_head,
                 mask_roi_extractor=mask_roi_extractor,
                 mask_head=mask_head,
                 shared_head=shared_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

        if nocs_head is not None:
            self.with_nocs = True
            self.nocs_head = build_head(nocs_head)
            self.nocs_roi_extractor = build_roi_extractor(nocs_roi_extractor)

        else:
            self.with_nocs =False

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
            nocs_pred = self.nocs_head(nocs_feats)

            nocs_targets = self.nocs_head.get_target(
                sampling_results, gt_coords, self.train_cfg)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_nocs = self.nocs_head.loss(nocs_pred, nocs_targets,
                                            pos_labels, mask_targets)
        return dict(loss_nocs=loss_nocs)


    def _shape_forward_train(self,
                            x,
                            sampling_results,
                            bbox_feats,
                            gt_voxels,
                            Ks,
                            **kwargs
                            ):

        losses = {}
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
            # with torch.no_grad():
            #     print(voxel_pred.all())
            # print(
            #       # f'voxel_feats: {voxel_feats[0, 0, 0, 0]}, '
            #       f'voxel_pred: {voxel_pred[0].all()}, '
            #       # f'gt_voxel: {gt_voxels[0][0][0]}')
            # )
            loss_voxel = self.voxel_head.loss(
                voxel_pred, sampling_results, gt_voxels, Ks)
            losses.update(dict(loss_voxel=loss_voxel))
        if self.with_mesh:
            init_mesh = self.voxel_head.init_mesh(voxel_pred)
            # if not self.with_voxel:
            #     if mesh_feats.size(0) > 0:
            #         init_mesh = ico_sphere(self.ico_sphere_level, mesh_feats.device)
            #         init_mesh = init_mesh.extend(mesh_feats.shape[0])
            #     else:
            #         init_mesh = Meshes(verts=[], faces=[])
            pred_meshes = self.mesh_head(mesh_feats, init_mesh)

            if not pred_meshes[0].isempty():
                loss_chamfer, loss_normals, loss_edge, target_meshes = self.mesh_head.loss(
                    Ks=Ks,
                    sampling_results=sampling_results,
                    pred_meshes=pred_meshes,
                    gt_meshes=kwargs['gt_meshes']
                )
                # if self._vis:
                #     self._misc["init_meshes"] = init_mesh
                #     self._misc["target_meshes"] = target_meshes
            else:
                loss_chamfer = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0
                loss_normals = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0
                loss_edge = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0

            losses.update(
                {
                    "loss_chamfer": loss_chamfer,
                    "loss_normals": loss_normals,
                    "loss_edge": loss_edge,
                }
            )
        return dict(loss_shape=losses)


    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,   # [(1000, 5)], len = bz
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                # assign_result 这里相当于 detectron2\modeling\roi_heads\roi_heads.py
                # label_and_sample_proposals函数下的 add_ground_truth_to_proposals (https://blog.csdn.net/qq_33891314/article/details/110789748)

                # 其意义在于 :
                # 当训练刚开始的时候，由于RPN网络参数是随机初始化的，导致生成的候选区域质量较差。
                # 这可能导致没有一个候选区域与真实框有足够大的交并比，作为正例参与第二阶段分类和回归头的训练。
                # 添加真实框到候选区域集合中，确保在最初训练的时候第二阶段有一定的正例。

                # 例如 proposal_list 为 [(1000, 5)], gt_bbox数目为 8, 则 assign_result 为  (1008, 5),
                # 注: gt_bbox被添加至 proposal_list的尾部(直接用了torch.cat(proposal, gt))
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
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        if self.with_nocs:
            nocs_results = self._nocs_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    mask_results['mask_targets'],
                                                    kwargs['gt_coords'])

            losses.update(nocs_results['loss_nocs'])
        if self.with_voxel:
            voxel_results = self._shape_forward_train(x=x,
                                                      sampling_results=sampling_results,
                                                      bbox_feats=bbox_results['bbox_feats'],
                                                      **kwargs
                                                      )
            losses.update(voxel_results['loss_shape'])

        if self.with_z:
            z_results = self._z_forward_train(x,
                                              sampling_results,
                                              bbox_results['bbox_feats'],
                                              kwargs['gt_zs'],
                                              gt_labels
                                              )
            losses.update(z_results['loss_z'])
        return losses


    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals = None,
                    rescale = False
        ):
        """Test without augmentation."""
        final_results = []
        assert self.with_bbox, "Bbox head must be implemented."
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]


        if self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)

        if self.with_voxel:
            shape_results = self.simple_test_shape(x,
                          img_metas,
                          det_bboxes,
                          rescale=False)

        return bbox_results, segm_results, shape_results


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


    def simple_test_shape(self,
                          x,
                          img_metas,
                          det_bboxes,
                          rescale=False):
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        shape_results = []
        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            shape_results = [[], []]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            rois = bbox2roi(_bboxes)
            voxel_feats = self.voxel_roi_extractor(
                x[:len(self.voxel_roi_extractor.featmap_strides)], rois)
            voxel_pred = self.voxel_head(voxel_feats)
            voxel_out = [i for i in voxel_pred.sigmoid().cpu().numpy()]
            shape_results.append(voxel_out)
            if self.with_mesh:
                mesh_feats = self.mesh_roi_extractor(x[:self.mesh_roi_extractor.num_inputs], rois)
                init_mesh = self.voxel_head.init_mesh(voxel_pred)
                mesh_pred = self.mesh_head(mesh_feats, init_mesh)
                mesh_pred = mesh_pred[-1] # 选取最后一个
                mesh_out = [(v, f) for (v, f) in zip(mesh_pred.verts_list(), mesh_pred.faces_list())]
                shape_results.append(mesh_out)
        return shape_results

    def simple_test_z(self,
                      x,
                      img_metas,
                      det_bboxes,
                      rescale=False
                      ):
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        z_results = []
        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            z_results = [[]]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            rois = bbox2roi(_bboxes)
            z_feats = self.z_roi_extractor(
                x[:len(self.z_roi_extractor.featmap_strides)], rois)
            z_pred = self.z_head(z_feats)
            z_results.append(z_pred)
        return z_results

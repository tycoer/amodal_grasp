from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.builder import build_head, HEADS, build_roi_extractor
from mmdet.core import bbox2roi
import torch
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


@HEADS.register_module()
class GraspROIHead(StandardRoIHead):
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
                 mesh_roi_extractor=None

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



    def _forward_voxel_head(self,
                            x,
                            sampling_results,
                            bbox_feats,
                            mask_targets,
                            ):

        losses = {}

        if not self.share_roi_extractor:
            pos_rois = bbox2roi(
                [res.pos_bboxes for res in sampling_results])
            voxel_feats = self.voxel_roi_extractor(x[:self.voxel_roi_extractor.num_inputs], pos_rois)
            mesh_feats = self.mesh_roi_extractor(x[:self.voxel_roi_extractor.num_inputs], pos_rois)

            if self.with_shared_head:
                voxel_feats = self.shared_head(voxel_feats)
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
            mesh_feats = bbox_feats[pos_inds]

        if voxel_feats.size(0) > 0:
            voxel_pred = self.voxel_head(voxel_feats)
            loss_voxel, target_voxels = voxel_rcnn_loss(
                voxel_pred, proposals, loss_weight=self.voxel_loss_weight
            )
            # voxel_targets = self.voxel_head.get_target(
            #     sampling_results, gt_coords, self.train_cfg)
            # pos_labels = torch.cat(
            #     [res.pos_gt_labels for res in sampling_results])
            # loss_voxel = self.voxel_head.loss(voxel_pred, voxel_targets,
            #                                 pos_labels, mask_targets)

            with torch.no_grad():
                vox_in = voxel_pred.sigmoid().squeeze(1)  # (N, V, V, V)
                init_mesh = cubify(vox_in, self.cubify_thresh)  # 1


        if mesh_feats.size(0) > 0:
            init_mesh = ico_sphere(self.ico_sphere_level, mesh_feats.device)
            init_mesh = init_mesh.extend(mesh_feats.shape[0])
        else:
            init_mesh = Meshes(verts=[], faces=[])
        pred_meshes = self.mesh_head(mesh_feats, init_mesh)

        loss_weights = {
            "chamfer": self.chamfer_loss_weight,
            "normals": self.normals_loss_weight,
            "edge": self.edge_loss_weight,
        }

        if not pred_meshes[0].isempty():
            loss_chamfer, loss_normals, loss_edge, target_meshes = mesh_rcnn_loss(
                pred_meshes,
                proposals,
                loss_weights=loss_weights,
                gt_num_samples=self.gt_num_samples,
                pred_num_samples=self.pred_num_samples,
                gt_coord_thresh=self.gt_coord_thresh,
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
        return losses






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
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            # print(bbox_results['loss_bbox'], img_metas[0]['scene_id'] )
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
                                                    kwargs['gt_coords'])

            losses.update(nocs_results['loss_nocs'])
        if self.with_voxel:
            voxel_results = self._voxel_forward_train(x)

        return losses


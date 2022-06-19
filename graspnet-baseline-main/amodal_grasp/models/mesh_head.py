# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
import fvcore.nn.weight_init as weight_init
import torch
# from detectron2.layers import ShapeSpec, cat
# from detectron2.utils.registry import Registry
# from pytorch3d.loss import chamfer_distance, mesh_edge_loss
# from pytorch3d.ops import GraphConv, SubdivideMeshes, sample_points_from_meshes, vert_align
# from pytorch3d.structures import Meshes
# from torch import nn
from torch.nn import functional as F
# from meshrcnn.structures.mesh import MeshInstances, batch_crop_meshes_within_box
from .utils.mesh import batch_crop_meshes_within_box
# ROI_MESH_HEAD_REGISTRY = Registry("ROI_MESH_HEAD")

from mmdet.models.builder import HEADS


# def mesh_rcnn_inference(pred_meshes, pred_instances):
#     """
#     Return the predicted mesh for each predicted instance
#
#     Args:
#         pred_meshes (Meshes): A class of Meshes containing B meshes, where B is
#             the total number of predictions in all images.
#         pred_instances (list[Instances]): A list of N Instances, where N is the number of images
#             in the batch. Each Instances must have field "pred_classes".
#
#     Returns:
#         None. pred_instances will contain an extra "pred_meshes" field storing the meshes
#     """
#     num_boxes_per_image = [len(i) for i in pred_instances]
#     pred_meshes = pred_meshes.split(num_boxes_per_image)
#
#     for pred_mesh, instances in zip(pred_meshes, pred_instances):
#         # NOTE do not save the Meshes object; pickle dumps become inefficient
#         if pred_mesh.isempty():
#             continue
#         verts_list = pred_mesh.verts_list()
#         faces_list = pred_mesh.faces_list()
#         instances.pred_meshes = MeshInstances([(v, f) for (v, f) in zip(verts_list, faces_list)])


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth, gconv_init="normal"):
        """
        Args:
          img_feat_dim: Dimension of features we will get from vert_align
          vert_feat_dim: Dimension of vert_feats we will receive from the
                        previous stage; can be 0
          hidden_dim: Output dimension for graph-conv layers
          stage_depth: Number of graph-conv layers to use
          gconv_init: How to initialize graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        # fc layer to reduce feature dimension
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        # deform layer
        self.verts_offset = nn.Linear(hidden_dim + 3, 3)

        # graph convs
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.verts_offset.weight)
        nn.init.constant_(self.verts_offset.bias, 0)

    def forward(self, x, mesh, vert_feats=None):
        img_feats = vert_align(x, mesh, return_packed=True, padding_mode="border")
        # 256 -> hidden_dim
        img_feats = F.relu(self.bottleneck(img_feats))
        if vert_feats is None:
            # hidden_dim + 3
            vert_feats = torch.cat((img_feats, mesh.verts_packed()), dim=1)
        else:
            # hidden_dim * 2 + 3
            vert_feats = torch.cat((vert_feats, img_feats, mesh.verts_packed()), dim=1)
        for graph_conv in self.gconvs:
            vert_feats_nopos = F.relu(graph_conv(vert_feats, mesh.edges_packed()))
            vert_feats = torch.cat((vert_feats_nopos, mesh.verts_packed()), dim=1)

        # refine
        deform = torch.tanh(self.verts_offset(vert_feats))
        mesh = mesh.offset_verts(deform)
        return mesh, vert_feats_nopos


@HEADS.register_module()
class MeshRCNNGraphConvHead(nn.Module):
    """
    A mesh head with vert align, graph conv layers and refine layers.
    """

    # self, cfg, input_shape: ShapeSpec,

    def __init__(self,
                 # model paras
                 input_channels=256,
                 num_stages=3,
                 num_graph_convs=3,
                 graph_conv_dim=128,
                 graph_conv_init='normal',
                 charmfer_loss_weight=1,
                 normals_loss_weight=0.1,
                 edge_loss_weight=1,

                 # other paras
                 gt_num_samples=5000,
                 pred_num_samples=5000,
                 gt_coord_thresh=5
                 ):
        super(MeshRCNNGraphConvHead, self).__init__()

        # # fmt: off
        # num_stages         = cfg.MODEL.ROI_MESH_HEAD.NUM_STAGES
        # num_graph_convs    = cfg.MODEL.ROI_MESH_HEAD.NUM_GRAPH_CONVS  # per stage
        # graph_conv_dim     = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_DIM
        # graph_conv_init    = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_INIT
        # input_channels     = input_shape.channels
        # # fmt: on

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            vert_feat_dim = 0 if i == 0 else graph_conv_dim
            stage = MeshRefinementStage(
                input_channels,
                vert_feat_dim,
                graph_conv_dim,
                num_graph_convs,
                gconv_init=graph_conv_init,
            )
            self.stages.append(stage)

        self.charmfer_loss_weight = charmfer_loss_weight
        self.normals_loss_weight = normals_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples
        self.gt_coord_thresh = gt_coord_thresh


    def forward(self, x, mesh):
        if x.numel() == 0 or mesh.isempty():
            return [Meshes(verts=[], faces=[])]

        meshes = []
        vert_feats = None
        for stage in self.stages:
            mesh, vert_feats = stage(x, mesh, vert_feats=vert_feats)
            meshes.append(mesh)
        return meshes

    def get_target(self,
                   sampling_results,
                   gt_meshes,
                   Ks):
        vert_targets, face_targets = [], []
        for sampling_result, mesh, K in zip(sampling_results, gt_meshes, Ks):
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            pos_proposals = sampling_result.pos_bboxes
            Ks_sampled = K.repeat(len(pos_assigned_gt_inds), 1)
            meshes_sampled = [mesh[i] for i in pos_assigned_gt_inds]
            mesh_targets = batch_crop_meshes_within_box(meshes=meshes_sampled,
                                                        boxes=pos_proposals,
                                                        Ks=Ks_sampled)
            target_meshes: Meshes
            vert_targets.extend(mesh_targets.verts_list())
            face_targets.extend(mesh_targets.faces_list())

        mesh_targets = Meshes(verts=vert_targets, faces=face_targets)
        return mesh_targets

    def loss(self,
             mesh_pred,
             mesh_targets):
        valid = mesh_targets.valid
        target_sampled_verts, target_sampled_normals = [], []
        for i in range(len(mesh_targets)):
            verts, normals = sample_points_from_meshes(
                mesh_targets[i], num_samples=self.gt_num_samples, return_normals=True
            )
            target_sampled_verts.append(verts)
            target_sampled_normals.append(normals)
            # if the rois are bad, the target verts can be arbitrarily large
            # causing exploding gradients. If this is the case, ignore the batch
            if self.gt_coord_thresh and verts.abs().max() > self.gt_coord_thresh:
                valid[i] = False

        target_sampled_verts = torch.cat(target_sampled_verts) # (num_box, self.gt_num_samples, 3)
        target_sampled_normals = torch.cat(target_sampled_normals) # # (num_box, self.gt_num_samples, 3)

        device = target_sampled_verts.device
        all_loss_chamfer = [torch.Tensor([0]).to(device)]
        all_loss_normals = [torch.Tensor([0]).to(device)]
        all_loss_edge = [torch.Tensor([0]).to(device)]
        for mesh in mesh_pred:
            if not torch.isfinite(mesh.verts_packed()).all():
                # pred_mesh 可能出现 inf 或者nan 导致 sample_points_from_meshes 报错
                continue
            pred_sampled_verts, pred_sampled_normals = sample_points_from_meshes(
                mesh, num_samples=self.pred_num_samples, return_normals=True
            )
            wts = (mesh.valid * valid).to(dtype=torch.float32)
            # chamfer loss
            loss_chamfer, loss_normals = chamfer_distance(
                pred_sampled_verts,
                target_sampled_verts,
                x_normals=pred_sampled_normals,
                y_normals=target_sampled_normals,
                weights=wts,
            )
            # chamfer loss
            all_loss_chamfer.append(loss_chamfer)
            # normal loss
            all_loss_normals.append(loss_normals)
            # mesh edge regularization
            loss_edge = mesh_edge_loss(mesh)
            all_loss_edge.append(loss_edge)

        loss_chamfer = sum(all_loss_chamfer) * self.charmfer_loss_weight
        loss_normals = sum(all_loss_normals) * self.normals_loss_weight
        loss_edge = sum(all_loss_edge) * self.edge_loss_weight
        return loss_chamfer, loss_normals, loss_edge

    # def loss(self,
    #          Ks,
    #          pred_meshes,
    #          gt_meshes,
    #          sampling_results
    #          ):
    #     gt_verts, gt_faces = [], []
    #     for sampling_result, voxel, K in zip(sampling_results, gt_meshes, Ks):
    #         pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
    #         pos_proposals = sampling_result.pos_bboxes
    #         Ks_sampled = [K[i] for i in pos_assigned_gt_inds]
    #         gt_meshes_sampled = [gt_meshes[i] for i in pos_assigned_gt_inds]
    #         target_meshes = batch_crop_meshes_within_box(meshes=gt_meshes_sampled,
    #                                                      boxes=pos_proposals,
    #                                                      Ks=Ks_sampled)
    #         target_meshes: Meshes
    #         gt_verts.extend(target_meshes.verts_list())
    #         gt_faces.extend(target_meshes.faces_list())
    #
    #     if len(gt_verts) == 0:
    #         return None, None
    #
    #     gt_meshes = Meshes(verts=gt_verts, faces=gt_faces)
    #     gt_valid = gt_meshes.valid
    #     gt_sampled_verts, gt_sampled_normals = sample_points_from_meshes(
    #         gt_meshes, num_samples=self.gt_num_samples, return_normals=True
    #     )
    #
    #     all_loss_chamfer = []
    #     all_loss_normals = []
    #     all_loss_edge = []
    #     for pred_mesh in pred_meshes:
    #         pred_sampled_verts, pred_sampled_normals = sample_points_from_meshes(
    #             pred_mesh, num_samples=self.pred_num_samples, return_normals=True
    #         )
    #         wts = (pred_mesh.valid * gt_valid).to(dtype=torch.float32)
    #         # chamfer loss
    #         loss_chamfer, loss_normals = chamfer_distance(
    #             pred_sampled_verts,
    #             gt_sampled_verts,
    #             x_normals=pred_sampled_normals,
    #             y_normals=gt_sampled_normals,
    #             weights=wts,
    #         )
    #         # chamfer loss
    #         loss_chamfer = loss_chamfer * self.charmfer_loss_weight
    #         all_loss_chamfer.append(loss_chamfer)
    #         # normal loss
    #         loss_normals = loss_normals * self.normals_loss_weight
    #         all_loss_normals.append(loss_normals)
    #         # mesh edge regularization
    #         loss_edge = mesh_edge_loss(pred_mesh)
    #         loss_edge = loss_edge * self.edge_loss_weight
    #         all_loss_edge.append(loss_edge)
    #
    #     loss_chamfer = sum(all_loss_chamfer)
    #     loss_normals = sum(all_loss_normals)
    #     loss_edge = sum(all_loss_edge)
    #
    #     # if the rois are bad, the target verts can be arbitrarily large
    #     # causing exploding gradients. If this is the case, ignore the batch
    #     if self.gt_coord_thresh and gt_sampled_verts.abs().max() > self.gt_coord_thresh:
    #         loss_chamfer = loss_chamfer * 0.0
    #         loss_normals = loss_normals * 0.0
    #         loss_edge = loss_edge * 0.0
    #
    #     return loss_chamfer, loss_normals, loss_edge, gt_meshes

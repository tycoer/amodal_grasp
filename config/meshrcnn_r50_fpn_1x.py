# model settings
# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            base_sizes=[32, 64, 128, 256, 512],  # tycoer
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0)),
    roi_head=dict(
        type='MeshRCNNROIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7,
                           # sampling_ratio=0,
                           sampling_ratio=2 #tycoer
                            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=9,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=0.0)),
        voxel_head=dict(type='VoxelRCNNConvUpsampleHead',
                        num_classes=9,
                        input_channels=256,
                        num_conv=256,
                        conv_dims=256,
                        num_depth=24,
                        reg_class_agnostic=True,
                        loss_weight=3
                        ),
        voxel_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=12, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        mesh_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mesh_head=dict(type='MeshRCNNGraphConvHead',
                       input_channels=256,
                       num_stages=3,
                       num_graph_convs=3,
                       graph_conv_dim=128,
                       graph_conv_init='normal',
                       charmfer_loss_weight=0.0,
                       normals_loss_weight=0.1,
                       edge_loss_weight=0.0
                       ),
        z_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        z_head=dict(type='FastRCNNFCHead',
                    input_channels=256 * 7 * 7,  # shape of z_feats
                    num_fc=2,
                    fc_dim=1024,
                    num_classes=9,
                    reg_class_agnostic=False,
                    loss_weight=0.0,
                    smooth_l1_beta=1.0
                    )
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# dataset settings
dataset_type = 'ConcatDataset'
data_root = 'data/nocs_data/'

img_norm_cfg = dict(
    # mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=True)
mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], to_rgb = True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPix3D', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Pix3DPipeline'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
                               'gt_voxels',
                               'gt_masks',
                               'Ks',
                               'gt_zs',
                               'gt_meshes',
                               ],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'item', 'img_norm_cfg')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'item', 'img_norm_cfg')
                 ),
        ])
]


data_root = '/disk2/data/pix3d'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(type="Pix3DDataset",
               data_root=data_root,
               anno_path=data_root + '/pix3d_s1_train.json',
               pipeline=train_pipeline),
    val=dict(type="Pix3DDataset",
             data_root=data_root,
             anno_path=data_root + '/pix3d_s1_test.json',
             pipeline=test_pipeline),
    test=dict(type="Pix3DDataset",
              data_root=data_root,
              anno_path=data_root + '/pix3d_s1_test.json',
              pipeline=test_pipeline),
)


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=1))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[5, 7, 9])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=11)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# find_unused_parameters=True
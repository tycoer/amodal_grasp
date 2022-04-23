# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='AmodalGrip',
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
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='NOCSROIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
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
            num_classes=6,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        nocs_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        nocs_head=dict(
            type='NOCSHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=6,
            num_bins=-1,
            norm_cfg=norm_cfg,
            loss_coord=dict(
                type='SymmetryCoordLoss', loss_weight=1.0)),
        ),
    grasp_head=dict(type='GraspHead'),
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
                num=64,
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
            mask_thr_binary=0.5))
)

# dataset settings
dataset_type = 'ConcatDataset'
data_root = 'data/nocs_data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

train_pipeline = [
                  dict(type='ResizeNOCS', min_dim=640, max_dim=640, padding=True, no_depth=True),
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='DefaultFormatBundleNOCS'),
                  dict(type='Collect', keys=['img', 'gt_labels', 'gt_bboxes', 'gt_masks', 'gt_coords',
                                             'gt_width', 'gt_qual', 'gt_rotations',
                                             'gt_occ',
                                             'voxel_grid', 'occ_points', 'gripper_T'],
                       meta_keys=['scene_id',
                                  # 'filename',
                                  # 'ori_shape',
                                  'img_shape',
                                  'pad_shape',
                                  'scale_factor',

                                  # 'flip',
                                  'img_norm_cfg',
                                  # 'domain'
                                                      ])]



# train_pipeline = [
#     dict(type='LoadColorAndDepthFromFile'),
#     dict(type='LoadAnnotationsNOCS', with_mask=True, with_coord=True),
#     # dict(type='ProcessDataNOCS'),
#     dict(type='ResizeNOCS', min_dim=480, max_dim=640, padding=True, no_depth=True),
#     dict(type='ExtractBBoxFromMask'),
#     dict(type='Normalize', **img_norm_cfg, is_nocs=True),
#     # dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundleNOCS'),
#     dict(type='Collect', keys=['img', 'gt_labels', 'gt_bboxes', 'gt_masks', 'gt_coords', 'scales'],
#          meta_keys=['img_path', 'depth_path', 'ori_shape', 'img_shape', 'pad_shape', 'window', 'scale_factor', 'domain']),
# ]
# train_pipeline2 = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
#          meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
#                     'scale_factor', 'flip', 'img_norm_cfg', 'domain')),
# ]
# test_pipeline = [
#     dict(type='LoadColorAndDepthFromFile'),
#     dict(
#         type='MultiSizeAugNOCS',
#         transforms=[
#             dict(type='ResizeNOCS', min_dim=480, max_dim=640, padding=False, no_depth=True),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img', 'depth']),
#             dict(type='Collect', keys=['img', 'depth'],
#                  meta_keys=['img_path', 'depth_path', 'ori_shape', 'img_shape', 'pad_shape',
#                             'window', 'scale_factor', 'domain', 'intrinsics', 'cat_ids', 'inst']),
#         ])
# ]


test_pipeline  = [
                  dict(type='ResizeNOCS', min_dim=640, max_dim=640, padding=True, no_depth=True),
                  # dict(type='Pad', size=(640, 640),),
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='DefaultFormatBundleNOCS'),
                  dict(type='Collect', keys=['img',
                                             # 'gt_labels', 'gt_bboxes', 'gt_masks', 'gt_coords',
                                             # 'gt_width', 'gt_qual', 'gt_rotations',
                                             # 'gt_occ',
                                             # 'voxel_grid', 'occ_points', 'gripper_T',
                                             ],
                       meta_keys=['scene_id',
                                  # 'filename',
                                  'ori_shape',
                                  'img_shape',
                                  'pad_shape',
                                  'scale_factor',

                                  # 'flip',
                                  'img_norm_cfg',
                                  # 'domain'
                                                      ])]



data_root = '/disk1/data/amodal_grip_dataset/data_test'


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(type="AmodalGripDataset",
               data_root=data_root,
               pipeline=train_pipeline),
    val=dict(type="AmodalGripDataset",
               data_root=data_root,
               pipeline=test_pipeline),
    test=dict(type="AmodalGripDataset",
               data_root=data_root,
               pipeline=test_pipeline),
    # train=dict(
    #     type=['NOCSCameraDataset', 'NOCSRealDataset', 'CocoDataset'],
    #     ann_file=[data_root + 'camera/val.txt',
    #               data_root + 'real/test.txt',
    #               data_root + 'coco/annotations/instances_train2017_nocs.json'
    #               ],
    #     img_prefix=[data_root + 'camera/val/',
    #                 data_root + 'real/real_test/',
    #                 data_root + 'coco/train2017/'
    #                 ],
    #     obj_model_dir=[data_root + 'obj_models/val/',
    #                    data_root + 'obj_models/real_test/',
    #                    None
    #                    ],
    #     intrinsics=[[[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]],
    #                 [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]],
    #                 None],
    #     sample_weight=[3, 1, 1],
    #     pipeline=[train_pipeline, train_pipeline, train_pipeline2],
    #     iters_per_epoch=1000),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_val2017.json',
    #     img_prefix=data_root + 'val2017/',
    #     pipeline=test_pipeline),
    # test=dict(
    #     type='NOCSRealDataset',
    #     ann_file=data_root + 'real/test_mini.txt',
    #     img_prefix=data_root + 'real/real_test/',
    #     obj_model_dir=data_root + 'obj_models/real_test/',
    #     intrinsics=[[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]],
    #     pipeline=test_pipeline))
)
# optimizer
optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[13, 13])
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
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/nocs_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters=True
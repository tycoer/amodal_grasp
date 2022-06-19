# model settings
norm_cfg = dict(type='BN', requires_grad=False)
model = dict(
    type='AmodalGrasp',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        in_channels=6,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
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
        type='StandardRoIHead',
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
            num_classes=80,
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
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
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
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)


keys = ['img', 'gt_masks', 'gt_bboxes', 'gt_labels',
        # heatmap
        'gt_heatmaps',
        # grasp_info
        'gt_gripper_quat', 'gt_gripper_T_uv_for_hm', 'gt_gripper_width', 'gt_gripper_qual',
        ]

train_pipeline = [
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='StackImgXYZ'),
                  dict(type='SimplePadding',
                       out_shape=(640, 640)),
                  dict(type='GenerateHM',
                       max_heatmap_num=20,
                       heatmap_shape=(40, 40),
                       sigma=5
                       ),
                  dict(type='WarpMask'),
                  dict(type='DefaultFormatBundle'),
                  dict(type='Collect', keys=keys,
                       meta_keys=['scene_id',
                                  'img_shape',
                                  'pad_shape',
                                  'scale_factor',
                                  'img_norm_cfg',
                                                      ])]

test_pipeline  = [
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='StackImgXYZ'),
                  dict(type='SimplePadding',
                       out_shape=(640, 640)),
                  dict(type='Transpose', keys=['img'], order=(2, 0, 1)),
                  dict(type='ToTensor', keys=['img']),
                  dict(type='Collect', keys=['img'],
                             meta_keys=['scene_id',
                                        'img_shape',
                                        # 'pad_shape',
                                        'scale_factor',
                                        'img_norm_cfg',
                                        ])
]



data_root = '/disk1/data/amodal_grasp/data_test/scenes_processed'


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(type="AmodalGraspDataset",
               data_root=data_root,
               pipeline=train_pipeline),
    val=dict(type="AmodalGraspDataset",
               data_root=data_root,
               pipeline=test_pipeline),
    test=dict(type="AmodalGraspDataset",
               data_root=data_root,
               pipeline=test_pipeline),
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
    step=[8, 18, 25])
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
total_epochs = 32
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/nocs_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
# find_unused_parameters=True
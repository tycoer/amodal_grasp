model = dict(type='AmodalGraspOnlyGrasp',
             backbone=dict(
                 type='ResNet',
                 in_channels=6,
                 depth=50,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 style='pytorch',
                 init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
             grasp_head=dict(type('GripHead')))


dataset_type = 'ConcatDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)


keys = ['img', 'gt_masks', 'gt_bboxes', 'gt_labels',
        # heatmap
        'gt_heatmaps',
        # grasp_info
        'gt_gripper_quat', 'gt_gripper_T_uv_for_hm', 'gt_gripper_width', 'gt_gripper_qual',
        # 'gt_gripper_valid_index'
        # 'grasp_map'
        ]

train_pipeline = [
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='StackImgXYZ'),
                  dict(type='SimplePadding',
                       out_shape=(320, 320)),
                  # dict(type='GenerateGraspMap', map_size=80),
                  dict(type='GenerateHM',
                       max_heatmap_num=10,
                       heatmap_shape=(80, 80),
                       kernal_size=15,
                       sigma_x=2
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
                       out_shape=(320, 320)),
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



data_root = '/disk1/data/amodal_grasp/packed_raw/scenes_processed'


data = dict(
    samples_per_gpu=4,
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
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[6, 8, 10])
checkpoint_config = dict(interval=6)
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
# find_unused_parameters=True
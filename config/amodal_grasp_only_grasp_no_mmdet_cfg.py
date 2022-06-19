model = dict(type='AmodalGraspOnlyGrasp',

             backbone=dict(
                 type='ResNet',
                 in_channels=3,
                 depth=50,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 style='pytorch',
             ),
                 # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
             grasp_head=dict(type='GraspHeadOnlyGrasp',
                             # loss_heatmap=dict(type='JointsMSELoss'),
                             loss_qual=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
                             loss_width=dict(type='MSELoss', loss_weight=1.0),
                             loss_quat=dict(type='MSELoss', loss_weight=1.0),
                             in_channels=512,
                             out_channels=32,
                             )
             )


dataset_type = 'ConcatDataset'

keys = ['img',
        'gt_grasp_uv_on_obj',
        'gt_grasp_vu_on_obj',
        'gt_grasp_qual',
        'gt_grasp_width',
        'gt_grasp_quat',
        ]

train_pipeline = [
                    # dict(type='SimplePadding', out_shape=(640, 640)),
                  dict(type='Transpose', keys=['img'], order=[2, 0, 1]),
                  dict(type='Collect', keys=keys, meta_keys=[])]

test_pipeline = [dict(type='SimplePadding', out_shape=(640, 640)),
                 dict(type='Transpose', keys=['img'], order=[2, 0, 1]),
                 dict(type='Collect', keys=['img'], meta_keys=[])]


data_root = '/disk1/data/giga/data_packed_train_raw'


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(type="AmodelGraspDatasetByGrasp",
               data_root=data_root,
               pipeline=train_pipeline),
    val=dict(type="AmodelGraspDatasetByGrasp",
               data_root=data_root,
               pipeline=test_pipeline),
    test=dict(type="AmodelGraspDatasetByGrasp",
               data_root=data_root,
               pipeline=test_pipeline),
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
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
find_unused_parameters=True
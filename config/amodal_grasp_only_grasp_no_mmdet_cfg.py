model = dict(type='AmodalGraspOnlyGraspNOMMDet',
             )


dataset_type = 'ConcatDataset'

keys = ['img',
        'gt_hm',
        # 'gt_hm_vu',
        # 'gt_uv',
        # 'gt_uv_obj',
        # 'gt_qual',
        # 'gt_width',
        # 'gt_quat',
        ]


meta_keys = [
        # 'gt_hm_vu',
        'gt_uv',
        'gt_uv_obj',
        'gt_qual',
        'gt_width',
        'gt_quat',
        ]



train_pipeline = [
                  dict(type='GenerateGlobalHM', hm_size=(30, 30)),
                  dict(type='Transpose', keys=['img'], order=[2, 0, 1]),
                  dict(type='Collect', keys=keys, meta_keys=[])]

test_pipeline = [
                # dict(type='SimplePadding', out_shape=(640, 640)),
                 dict(type='Transpose', keys=['img'], order=[2, 0, 1]),
                 dict(type='Collect', keys=['img'], meta_keys=meta_keys)]


data_root = '/disk1/data/giga/data_packed_train_raw'


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(type="AmodalGraspDatasetByScene",
               data_root=data_root,
               pipeline=train_pipeline),
    val=dict(type="AmodalGraspDatasetByScene",
               data_root=data_root,
               pipeline=test_pipeline),
    test=dict(type="AmodalGraspDatasetByScene",
               data_root=data_root,
               pipeline=test_pipeline),
)
# optimizer
optimizer = dict(type='SGD', lr=0.02,
                 momentum=0.9,
                 weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10, 15, 18])
checkpoint_config = dict(interval=5)
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
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/nocs_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters=True
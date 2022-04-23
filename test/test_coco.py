from mmdet.datasets.coco import CocoDataset

dataset_type = 'CocoDataset'
data_root = '/disk1/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]


if __name__ == '__main__':
    dataset = CocoDataset(data_root='/disk1/data/coco',
                          ann_file=data_root + 'annotations/instances_val2017.json',
                          img_prefix=data_root + 'val2017/',
                          pipeline=train_pipeline)
    data = dataset[0]

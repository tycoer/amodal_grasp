from model_2d.dataset_by_grasp import AmodelGraspDatasetByGrasp


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


keys = ['img', 'gt_masks', 'gt_bboxes', 'gt_labels',
        'gripper_T_uv_on_obj', 'gripper_quat',
        'gripper_width', 'gripper_qual'
        ]

train_pipeline = [
                  dict(type='Normalize', **img_norm_cfg),
                  dict(type='StackImgXYZ'),
                  dict(type='SimplePadding',
                       out_shape=(320, 320)),
                  dict(type='NormalizeGraspUV'),
                  # dict(type='GenerateHM',
                  #      max_heatmap_num=20,
                  #      heatmap_shape=(40, 40),
                  #      sigma=5
                  #      ),
                  dict(type='WarpMask'),
                  dict(type='DefaultFormatBundle'),
                  dict(type='Collect', keys=keys,
                       meta_keys=['scene_id',
                                  'img_shape',
                                  'pad_shape',
                                  'scale_factor',
                                  'img_norm_cfg',
                                                      ])]

if __name__ == '__main__':
    dataset = AmodelGraspDatasetByGrasp(data_root='/disk1/data/amodal_grasp/packed_raw',
                                        pipeline=train_pipeline)
    res = dataset[0]
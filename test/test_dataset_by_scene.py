from model_2d.dataset_by_scene import AmodalGraspDatasetByScene
from mmdet.models.roi_heads.mask_heads
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
                  dict(type='GenerateGlobalHM', hm_size=(32, 32)),
                  dict(type='Transpose', keys=['img'], order=[2, 0, 1]),
                  dict(type='Collect', keys=keys, meta_keys=[])]

test_pipeline = [
                # dict(type='SimplePadding', out_shape=(640, 640)),
                 dict(type='Transpose', keys=['img'], order=[2, 0, 1]),
                 dict(type='Collect', keys=['img'], meta_keys=meta_keys)]

if __name__ == '__main__':

    dataset = AmodalGraspDatasetByScene(data_root='/disk1/data/giga/data_packed_train_raw',
                                        pipeline=train_pipeline)
    data = dataset[0]
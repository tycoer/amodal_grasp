from amodal_grasp.dataset.amodal_grasp_dataset import GraspNetDatasetForAmodalGrasp
import time
train_pipeline = [
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
         meta_keys=['pad_shape']),
]



if __name__ == '__main__':
    dataset = GraspNetDatasetForAmodalGrasp(data_root='/disk2/data/graspnet',
                                            )
    data1 = dataset[9150]
    data2 = dataset[14993]
    data3 = dataset[12774]
    data4 = dataset[18500]
    data5 = dataset[21316]

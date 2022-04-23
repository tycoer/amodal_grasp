import os
import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
from meshrcnn.structures import MeshInstances, VoxelInstances
from detectron2.structures import Boxes, BoxMode, Instances


class AmodalGraspDataset(CustomDataset):
    CLASSES = {'bottle': '0',
             'bowl': '1',
             'can': '2',
             'cap': '3',
             'cell_phone': '4',
             'mug': '5'}

    def __init__(self,
                 data_root,
                 pipeline=None,
                 **kwargs):
        self.data_root = data_root
        self.scene_id_list = os.listdir(self.data_root)
        self.scene_abs_path = tuple(os.path.join(self.data_root, i) for i in self.scene_id_list)

        # self.nocs_para_path = os.path.join(self.data_root, 'nocs_para.json')
        # with open(self.nocs_para_path, 'r') as f:
        #     self.nocs_para = json.load(f)
        self.mesh_processed_path = os.path.join(self.data_root, 'mesh.npz')
        self.gt_mesh_dict = dict(np.load(self.mesh_processed_path))


        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self), dtype='int64')

    def __len__(self):
        return len(self.scene_id_list)


    def __getitem__(self, item):
        scene_abs_path = self.scene_abs_path[item]
        results = dict(np.load(scene_abs_path))

        # add mesh_info
        obj_names = results['obj_names']



        # add other info
        results['item'] = item
        results['scene_id'] = self.scene_id_list[item]
        results['img_shape'] = results['img'].shape
        results['gt_labels'] = np.int64(results['gt_labels'])


        if self.pipeline is not None:
            results = self.pipeline(results)
        return results


    def get_mesh_and_voxel(self, obj_names):
        voxel = []
        mesh = []
        for i in obj_names:
            voxel.append(self.gt_mesh_dict[i]['gt_voxel'])
            mesh.append(self.gt_mesh_dict[i]['gt_mesh'])
        return voxel, mesh



def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width

    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        masks = [obj["segmentation"] for obj in annos]
        target.gt_masks = torch.stack(masks, dim=0)

    # camera
    if len(annos) and "K" in annos[0]:
        K = [torch.tensor(obj["K"]) for obj in annos]
        target.gt_K = torch.stack(K, dim=0)

    if len(annos) and "voxel" in annos[0]:
        voxels = [obj["voxel"] for obj in annos]
        target.gt_voxels = VoxelInstances(voxels)

    if len(annos) and "mesh" in annos[0]:
        meshes = [obj["mesh"] for obj in annos]
        target.gt_meshes = MeshInstances(meshes)

    if len(annos) and "dz" in annos[0]:
        dz = [obj["dz"] for obj in annos]
        target.gt_dz = torch.tensor(dz)

    return target




def dataset_bridge_to_meshrcnn(results):
    height, width = results['img'].shape[:2]
    image = to_tensor(results['img'][:, :, :3]).float().transpose(2, 0, 1)

    # get instance
    targets = Instances((height, width))
    targets.gt_classes = torch.int32(results['gt_labels'])
    targets.gt_boxes = Boxes(results['gt_boxes'])
    targets.gt_masks = torch.float32(results['gt_masks'])
    # targets.gt_K = torch.stack(K, dim=0)
    targets.gt_voxels = VoxelInstances(results['gt_voxel'])
    targets.gt_meshes = MeshInstances(results['gt_meshes'])
    # targets.gt_dz =






    results_meshrcnn = dict(image=image,
                            height=height,
                            width=width,
                            image_id=results['item'],
                            instance=instance,
                            )

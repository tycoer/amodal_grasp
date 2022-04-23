import os
import numpy as np
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.builder import DATASETS
import tqdm
import json
import pandas as pd
from scipy.spatial.transform import Rotation
from mmdet.datasets import CustomDataset
import glob


def pc_cam_to_pc_world(pc, extrinsic):
    # 点云相机坐标系到世界坐标系的转换
    # pc shape (n , 3)
    # extrinsic shape (1, 7) 前四个数是四元数, 后三个数是平移
    Q = extrinsic[0, :4]
    T = extrinsic[0, 4:]
    extr = np.eye(4)
    extr[:3, :3] = Rotation.from_quat(Q).as_matrix()
    extr[:3, 3] = T

    extr_inv = np.linalg.inv(extr)

    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ pc.T).T + T # 注意 R 需要左乘, 右乘会得到错误结果
    return pc



def depth2pc(depth, fx, fy, cx, cy, w, h, depth_scale=1, ):
    h_grid, w_grid= np.mgrid[0: h, 0: w]
    z = depth / depth_scale
    x = (w_grid - cx) * z / fx
    y = (h_grid - cy) * z / fy
    xyz = np.dstack((x, y, z))
    return xyz


def extract_box_from_mask(mask):
    # mask shape(480, 640)
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    box = np.array([x1, y1, x2, y2])
    return box


@DATASETS.register_module()
class AmodalGripDataset(CustomDataset):
    CLASSES = {'bottle': '0',
             'bowl': '1',
             'can': '2',
             'cap': '3',
             'cell_phone': '4',
             'mug': '5'}

    def __init__(self,
                 data_root,
                 pipeline=None,
                 with_grip=True,
                 with_reconstruction=True,
                 **kwargs):
        self.data_root = data_root
        self.anno_root = os.path.join(self.data_root, 'mesh_pose_list')
        self.scene_root = os.path.join(self.data_root, 'scenes')
        self.setup_path = os.path.join(self.data_root, 'setup.json')
        self.nocs_path = os.path.join(self.data_root, 'nocs_para.json')
        self.grip_path = os.path.join(self.data_root, 'grasps.csv')
        self.occ_root = os.path.join(self.data_root, 'occ')
        self.voxel_grid_root = os.path.join(self.data_root, 'voxel_grid')
        # read files
        self.df = pd.read_csv(self.grip_path)
        with open(self.setup_path, 'r') as f:
            setup = json.load(f)
        with open(self.nocs_path, 'r') as f:
            self.nocs_para = json.load(f)

        self.h = 480
        self.w = 640
        self.K = setup['intrinsic']['K']
        self.fx, self.fy, self.cx, self.cy = self.K[0], self.K[4], self.K[2], self.K[5]
        self.camera = dict(h=self.h,
                           w=self.w,
                           fx=self.fx,
                           fy=self.fy,
                           cx=self.cx,
                           cy=self.cy)

        self.size = setup['size']

        self.with_grip = with_grip

        self.with_reconstruction = with_reconstruction

        self.image_info = self.preprocess_images()

        self.pipeline = pipeline
        if self.pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self))

    def __len__(self):
        return len(self.grip.index)


    def preprocess_images(self):
        results = {}
        for i in tqdm.tqdm(os.listdir(self.scene_root)):
            anno_abs_path = os.path.join(self.anno_root, i[:-4] + '.npy')
            image_abs_path = os.path.join(self.scene_root, i)
            images = dict(np.load(image_abs_path, allow_pickle=True).items())
            annos = list(np.load(anno_abs_path, allow_pickle=True))
            rgb = images['rgb_imgs'][0]
            depth = images['depth_imgs'][0]
            mask = images['mask_imgs'][0]
            extrinsic = images['extrinsics']

            xyz = depth2pc(depth, depth_scale=1, **self.camera) #xyz shape (480, 640, 3)
            # xyz 转换成世界坐标系的原因: 仿真器中记录的 物体的pose, gripper的pose都是世界坐标系下的
            # 而 xyz 在相机坐标系下, 故转为世界坐标系以统一
            xyz_world = pc_cam_to_pc_world(xyz.reshape(-1, 3), extrinsic)
            masks = []
            nocs_maps = []
            labels = []
            boxes_2d = []
            num_instance = len(annos)
            for anno in annos:
                ###############  mask ##############
                uid = anno['uid']
                mask_obj = mask==uid # 单个物体的mask
                masks.append(mask_obj)
                # mask中
                # -1：sky
                # 0：plane
                # 1：gripper # 生成数据时被移除
                # 故 索引物体的mask应从2开始

                # anno 从uid=2开始
                ###############boxes_2d #############
                box_2d = extract_box_from_mask(mask_obj)
                boxes_2d.append(box_2d)

                ###############  nocs #############
                xyz_obj = xyz_world.copy()
                scale = anno['scale']
                H = anno['pose']
                H_inv = np.linalg.inv(H)
                R = H_inv[:3, :3]
                T = H_inv[:3, 3]

                xyz_obj = (R @ xyz_obj.T).T + T # 将物体的pose还原到原点
                xyz_obj = xyz_obj / scale # 将物体还原为原始大小
                # xyz_obj[~mask_obj.flatten()] = 0 # 单个物体的partical点云, 非该物体的xyz置0
                nocs_para_obj = self.nocs_para[anno['obj_name']] # 物体的 nocs 参数
                norm_factor, norm_corner = nocs_para_obj['norm_factor'], np.array(nocs_para_obj['norm_corner'])
                nocs_map = (xyz_obj - norm_corner[0]) * norm_factor + np.array((0.5, 0.5, 0.5)).reshape(1, 3) - 0.5 * (norm_corner[1] - norm_corner[0]) * norm_factor
                nocs_map[~mask_obj.flatten()] = 0
                nocs_maps.append(nocs_map)
                ################ labels ###########
                labels.append(self.CLASSES[anno['category']])

            nocs_maps = np.stack(nocs_maps, axis=-2).reshape(self.h, self.w, num_instance, 3)
            masks = np.stack(masks, axis=-1)
            boxes_2d = np.int32(boxes_2d)
            labels = np.int32(labels)

            results[os.path.basename(i)[:-4]] = dict(
                                                gt_masks=masks, # shape (480, 640, num_instance)
                                                gt_coords=nocs_maps, # shape (480, 640, num_instance, 3)
                                                img=rgb,
                                                gt_labels=labels,
                                                gt_bboxes=boxes_2d
                                                )
        return results


    def __getitem__(self, item):
        scene_id = self.df.loc[item, "scene_id"]

        # grasps
        ori = Rotation.from_quat(self.df.loc[item, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[item, "x":"z"].to_numpy(np.single)
        width = self.df.loc[item, "width"].astype(np.single)
        label = self.df.loc[item, "label"].astype(np.long)
        pos = pos / self.size - 0.5
        width = width / self.size

        # voxel_grid
        voxel_grid = dict(np.load(os.path.join(self.voxel_grid_root, scene_id+'.npz')))['grid'][0]



        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        occ_points, occ = self.read_occ(scene_id, num_point=2048)
        occ_points = occ_points / self.size - 0.5


        result = dict(
             # input
             voxel_grid=voxel_grid,
             occ_points = occ_points,
             occ = occ,
             gripper_T = pos,
             # gt
             gt_width=np.array(width),
             gt_label=np.array(label),
             gt_rotations=rotations,
             )


        image_info = self.image_info[scene_id]
        result.update(image_info)
        if self.pipeline is not None:
            result = self.pipeline(result)
        return result


    def __len__(self):
        return len(self.df)


    def read_occ(self, scene_id, num_point):
        occ_paths = glob.glob(os.path.join(self.occ_root, scene_id, '*.npz'))
        path_idx = np.random.randint(0, len(occ_paths), dtype=int)
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ


def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]



if __name__ == '__main__':
    data_root = '/home/hanyang/amodal_grisp/data_test'
    dataset = AmodalGripDataset(data_root=data_root)
    data = dataset[0]
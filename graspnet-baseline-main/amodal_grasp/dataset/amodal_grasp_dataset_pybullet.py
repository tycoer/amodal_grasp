import os
import numpy as np
import cv2
from torchvision.transforms.functional import normalize, to_tensor
from torchvision.transforms import ToTensor
from torchvision.ops.boxes import clip_boxes_to_image
import json
from glob import glob
from tqdm import tqdm
import time
from os.path import join
import torch
import time
import matplotlib.pyplot as plt
import trimesh

def resize_box(box_xyxy, ori_wh, resized_wh):
    scale_factor_xyxy = ((np.array(ori_wh) / np.array(resized_wh))[None, :]).repeat(2, 1)
    box_xyxy_resized = box_xyxy / scale_factor_xyxy
    return box_xyxy_resized

class DepthToPc:
    def __init__(self,
                 depth_size=(720, 1280),
                 depth_scale=1000):
        self.depth_size = depth_size
        self.h, self.w = depth_size
        self.h_grid, self.w_grid = np.mgrid[0: self.h, 0: self.w]
        self.depth_scale = depth_scale
    def __call__(self, depth, fx, fy, cx, cy):
        z = depth / self.depth_scale
        x = (self.w_grid - cx) * z / fx
        y = (self.h_grid - cy) * z / fy
        xyz = np.dstack((x, y, z))
        return xyz

class AmodalGraspDataset:
    def __init__(self,
                 data_root,
                 split='train',
                 with_mesh=True,
                 with_pc=True,
                 with_nocs=True,
                 with_grasp=True,
                 img_norm_std=[0.229,0.224,0.225],
                 img_norm_mean=[0.485,0.456,0.406],
                 norm_quat=True,
                 ):

        self.obj_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
                   12, 13, 14, 15, 34, 37, 43,
                   46, 58, 61, 65,
                   ]

        self.norm_quat = norm_quat
        self.split = split
        self.data_root = data_root
        self.with_grasp = with_grasp
        self.with_nocs = with_nocs
        self.with_pc = with_pc
        self.with_mesh = with_mesh
        self.img_norm_std = img_norm_std
        self.img_norm_mean = img_norm_mean




        anns_root = join(self.data_root, 'scenes/ann')
        rgbs_root = join(self.data_root, 'scenes/rgb')
        segs_root = join(self.data_root, 'scenes/seg')
        depths_root = join(self.data_root, 'scenes/depth')
        nocs_root = join(self.data_root, 'scenes/nocs')
        grasp_root = join(self.data_root, 'scenes/grasp')

        self.rgbs_path = sorted(glob(rgbs_root + '/*.jpg'))
        self.segs_path = sorted(glob(segs_root + '/*.png'))
        self.depths_path = sorted(glob(depths_root + '/*.png'))
        self.anns_path = sorted(glob(anns_root + '/*.json'))
        self.nocs_path = sorted(glob(nocs_root + '/*.jpg'))
        self.grasps_path = sorted(glob(grasp_root + '/*.npz'))

        self.setup_path = join(data_root, 'setup.json')
        with open(self.setup_path, 'r') as f:
            self.setup = json.load(f)
        intr = self.setup['intrinsic']
        self.H, self.W, self.K = intr['width'], intr['height'], intr['K']
        self.intrinsics = np.float32(self.K).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = self.K[0], self.K[4], self.K[2], self.K[5]

        self.depth2pc = DepthToPc(depth_size=(self.H, self.W), depth_scale=1000)
        self.pipeline = ToTensor()

        if self.with_mesh:
            self.mesh_dict = {}
            self.voxel_dict = {}
            models_root = join(data_root, 'models_collision')
            for i in self.obj_ids:
                mesh_name = f'{str(i).zfill(3)}.obj'
                mesh_path = join(models_root, mesh_name)
                mesh = trimesh.load(mesh_path)

                voxel_name = f'{str(i).zfill(3)}.binvox'
                voxel_path = join(models_root, voxel_name)
                voxel = trimesh.load(voxel_path)
                self.voxel_dict[i] = voxel.points

                mesh: trimesh.Trimesh
                self.mesh_dict[i] = (mesh.vertices, mesh.faces)


        self.CLASSES = [str(i) for i in self.obj_ids]
        self.label_map = {obj_id: i for i, obj_id in enumerate(self.obj_ids)}

        self.split_ratio = 0.8
        self.split_num = int(len(self.rgbs_path) * self.split_ratio)
        if self.split == 'train':
            self.rgbs_path = self.rgbs_path[:self.split_num]
            self.depths_path = self.depths_path[:self.split_num]
            self.segs_path = self.segs_path[:self.split_num]
            self.anns_path = self.anns_path[:self.split_num]
            self.nocs_path = self.nocs_path[:self.split_num]
            self.grasps_path = self.grasps_path[:self.split_num]
        else:
            self.rgbs_path = self.rgbs_path[self.split_num:]
            self.depths_path = self.depths_path[self.split_num:]
            self.segs_path = self.segs_path[self.split_num:]
            self.anns_path = self.anns_path[self.split_num:]
            self.nocs_path = self.nocs_path[self.split_num:]
            self.grasps_path = self.grasps_path[self.split_num:]


    def __getitem__(self, item):
        # start = time.time()
        rgb_path, depth_path, seg_path, ann_path, nocs_path, grasp_path = (self.rgbs_path[item],
                                                                           self.depths_path[item],
                                                                           self.segs_path[item],
                                                                           self.anns_path[item],
                                                                           self.nocs_path[item],
                                                                           self.grasps_path[item])

        # rgb
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        img = rgb
        # ann
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        # mask, label and box
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        seg_ids= np.unique(seg)[1:]
        num_instance = len(seg_ids)
        box_xyxy, labels, obj_scales, obj_poses = [], [], [], []
        for i in seg_ids:
            box_xyxy.append(ann[str(i)]['box_xyxy'])
            labels.append(self.label_map[int(ann[str(i)]['mesh_path'].split('/')[-2])])
            obj_scales.append(ann[str(i)]['scale'])
            obj_poses.append(np.float32(ann[str(i)]['pose_cam']))
        box_xyxy = np.float32(box_xyxy)
        labels = np.int64(labels)
        obj_poses = np.float32(obj_poses)

        mask = (seg != 0).astype('int32')
        maps = mask[:, :, None]

        if self.with_pc:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            pc = self.depth2pc(depth, self.fx, self.fy, self.cx, self.cy)
            img = np.dstack((img, pc))

        if self.with_nocs:
            nocs = cv2.imread(nocs_path, cv2.IMREAD_UNCHANGED)
            nocs = np.float32(nocs / 255) # decode
            maps = np.dstack((maps, nocs))

        if self.with_grasp:
            # 耗时约 0.2s
            grasp = np.load(grasp_path)['grasp_map']
            maps = np.dstack((maps, grasp))

        if self.with_mesh:
            meshes, voxels = [], []
            for i in seg_ids:
                ann_ = ann[str(i)]
                pose_cam = np.float32(ann_['pose_cam'])
                obj_name = int(ann_['mesh_path'].split('/')[-2])
                verts, faces = self.mesh_dict[obj_name]
                verts = verts.copy()
                vox = self.voxel_dict[obj_name].copy()
                verts = verts * ann_['scale']
                vox = vox * ann_['scale']
                verts_cam = (pose_cam[:3, :3] @ verts.T).T + pose_cam[:3, 3]
                vox_cam = (pose_cam[:3, :3] @ vox.T).T + pose_cam[:3, 3]

                vox_cam *= np.float32([[-1, -1, 1]])
                verts_cam *= np.float32([[-1, -1, 1]])

                voxels.append(torch.from_numpy(vox_cam).float())
                meshes.append((torch.from_numpy(verts_cam).float(), torch.from_numpy(faces).int()))

        '''
        img_show = img[:, :, :3].astype('uint8')
        plt.imshow(img_show)
        plt.show()
        for i in np.int32(box_xyxy):
            cv2.rectangle(img_show, i[:2], i[2:], (255, 0, 0))
            plt.imshow(img_show)
            plt.show()
            
        for i, vox in enumerate(voxels):
            print(vox.shape)
            _ = trimesh.PointCloud(vox).export(f'/home/guest/{i}.ply')
        _ = trimesh.PointCloud(pc.reshape(-1, 3)).export('/home/guest/pc.ply')
        '''

        gt_maps = np.zeros((num_instance,) + maps.shape)
        for i, seg_id in enumerate(seg_ids):
            gt_maps[i][seg == seg_id] = maps[seg == seg_id]


        '''
        plt.imshow(rgb)
        plt.show()
        '''

        # to tensor
        rgb = to_tensor(img[:, :, :3].astype('uint8'))
        rgb = normalize(rgb, std=self.img_norm_std, mean=self.img_norm_mean)
        pc = torch.from_numpy(img[:, :, 3:6].transpose(2, 0, 1))
        img = torch.cat((rgb, pc))
        labels = torch.from_numpy(labels)
        box_xyxy = torch.from_numpy(box_xyxy)
        box_xyxy = clip_boxes_to_image(box_xyxy, size=(self.H, self.W))
        if self.norm_quat:
            gt_maps[:, :, :, 5:9] = ((gt_maps[:, :, :, 5:9] + 1) / 2) * gt_maps[:, :, :, 4:5]

        results = dict(
                       img=img.float(),
                       depth=depth / 1000,
                       gt_bboxes=box_xyxy.float(),
                       gt_labels=labels.long(),
                       gt_masks=gt_maps[:, :, :, 0],
                       gt_nocs=gt_maps[:, :, :, 1:4],
                       gt_grasps =gt_maps[:, :, :, 4:],
                       gt_meshes = meshes,
                       gt_voxels = voxels,
                       filename=rgb_path,
                       Ks=torch.Tensor([[self.fx, self.cx, self.cy]]).float(),
                       intrinsics=self.intrinsics,
                       obj_poses=obj_poses,
                       obj_scales=obj_scales
                       )


        return results

    def __len__(self):
        return len(self.rgbs_path)


if __name__ == '__main__':
    import time
    dataset = AmodalGraspDataset(data_root='/disk2/data/graspnet/amodal_grasp/pybullet')
    start = time.time()
    data = dataset[0]
    print(f'data time {time.time() - start}')
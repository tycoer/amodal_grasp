import os
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import json
import pandas as pd
import open3d as o3d
from sklearn.neighbors import KDTree


def xyz2uv(xyz, fx, fy, cx, cy):
    xyz=np.array(xyz).flatten()
    x,y=xyz[0]/xyz[2],xyz[1]/xyz[2]
    u,v=x*fx+cx,y*fy+cy
    uv=np.int0([u,v])
    return uv


def process_grasp(csv_path):
    csv = pd.read_csv(csv_path)
    scene_id_unique = csv['scene_id'].unique()
    grasp_infos = {i: [] for i in scene_id_unique}
    for i in tqdm.tqdm(range(csv.__len__())):
        grasp_info = csv.iloc[i].to_list()
        grasp_infos[grasp_info[0]].append(grasp_info[1:])
    return grasp_infos

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


def get_extr(extrinsic):
    Q = extrinsic[0, :4]
    T = extrinsic[0, 4:]
    extr = np.eye(4)
    extr[:3, :3] = Rotation.from_quat(Q).as_matrix()
    extr[:3, 3] = T
    return extr 

def grasp_world_to_cam(grasp_info, extr, intr):
    
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    
    
    
    grasp_info = np.array(grasp_info)
    grasp_quat = np.float32(grasp_info[:, :4])
    grasp_T = np.float32(grasp_info[:, 4:7])
    grasp_width = np.float32(grasp_info[:, 7])
    grasp_qual = np.int32(grasp_info[:, 8])
    
    quat_cam_list = []
    T_cam_list = []
    T_uv_list = []
    H_list = []
    for quat , T in zip(grasp_quat, grasp_T):
        H_grasp_world = np.eye(4)
        H_grasp_world[:3, :3] = Rotation.from_quat(quat).as_matrix()
        H_grasp_world[:3, 3] = T
        
        H_grasp_cam = extr.dot(H_grasp_world)
        quat_cam = Rotation.from_matrix(H_grasp_cam[:3, :3]).as_quat()
        T_cam = H_grasp_cam[:3, 3]
        T_uv = xyz2uv(T_cam, fx, fy, cx, cy)
        T_uv_list.append(T_uv)
        T_cam_list.append(T_cam)
        quat_cam_list.append(quat_cam)
        H_list.append(H_grasp_cam)
    grasp_T_cam = np.array(T_cam_list)
    grasp_quat_cam = np.array(quat_cam_list)
    grasp_T_uv =np.array(T_uv_list)    
    grasp_H_cam = np.array(H_list)
    return grasp_quat_cam, grasp_T_cam, grasp_T_uv, grasp_width, grasp_qual, grasp_H_cam
    
        
def project_grasp_on_object(grasp_T_cam, obj_xyz):
    # 将位于物体外的抓点投射到物体表面
    knn = KDTree(obj_xyz)
    distances, ids = knn.query(grasp_T_cam, k=1)
    grasp_T_cam_on_object = obj_xyz[ids]
    return grasp_T_cam_on_object



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


def preprocess_images(data_root):
    # path init
    scene_root = os.path.join(data_root, 'scenes')
    anno_root = os.path.join(data_root, 'mesh_pose_list')
    setup_path = os.path.join(data_root, 'setup.json')
    nocs_path = os.path.join(data_root, 'nocs_para.json')
    save_dir = os.path.join(data_root, 'scenes_processed')
    csv_path = os.path.join(data_root, 'grasps.csv')

    # file read
    os.makedirs(save_dir, exist_ok=True)
    with open(setup_path, 'r') as f:
        setup = json.load(f)
    with open(nocs_path, 'r') as f:
        nocs_para = json.load(f)
    K = setup['intrinsic']['K']

    w, h = int(K[2] * 2), int(K[5] * 2)
    camera = dict(h=h,
                  w=w,
                  fx=K[0],
                  fy=K[4],
                  cx=K[2],
                  cy=K[5])


    CLASSES = {'bottle': '0',
               'bowl': '1',
               'can': '2',
               'cap': '3',
               'cell_phone': '4',
               'mug': '5'}
    print('Processing grasp...')
    grasp_infos = process_grasp(csv_path=csv_path)

    results = {}
    for i in tqdm.tqdm(os.listdir(scene_root)):
        scene_id = i[:-4]
        anno_abs_path = os.path.join(anno_root, scene_id + '.npy')
        image_abs_path = os.path.join(scene_root, i)
        images = dict(np.load(image_abs_path, allow_pickle=True).items())
        annos = list(np.load(anno_abs_path, allow_pickle=True))
        rgb = images['rgb_imgs'][0]
        depth = images['depth_imgs'][0]
        mask = images['mask_imgs'][0]
        extrinsic = images['extrinsics']

        xyz = depth2pc(depth, depth_scale=1, **camera)  # xyz shape (480, 640, 3)
        extr =  get_extr(extrinsic)
        uid_mask = np.unique(mask)
        # xyz 转换成世界坐标系的原因: 仿真器中记录的 物体的pose, gripper的pose都是世界坐标系下的
        # 而 xyz 在相机坐标系下, 故转为世界坐标系以统一
        # xyz_world = pc_cam_to_pc_world(xyz.reshape(-1, 3), extrinsic)

        ######### grasps processing #############
        if scene_id not in grasp_infos:
            continue
        grasp_info = grasp_infos[scene_id]
        grasp_quat_cam, grasp_T_cam, grasp_T_uv, grasp_width, grasp_qual, grasp_H_cam = grasp_world_to_cam(grasp_info, extr, camera)
        # gripped_obj_uid_list = [[i, info[-1]] for i, info in enumerate(grasp_info) if info[-1] != "None"]
        # if len(gripped_obj_uid_list) == 0:
        #     # 如果一个scene中没有有效的抓取则跳过整个scene
        #     continue
        # grasp_T_cam_on_obj_list = []
        # grasp_T_uv_on_obj_list = []
        # for j, uid in gripped_obj_uid_list:
        #     uid = int(uid)
        #     if uid in uid_mask:
        #         mask_obj = mask == uid
        #         xyz_obj = xyz.reshape(-1, 3)[mask_obj.flatten()]
        #         knn = KDTree(xyz_obj)
        #         _, id = knn.query(grasp_T_cam[j][None], k=1)
        #         id = id.flatten()
        #         grasp_T_cam_on_obj = xyz_obj[id]
        #         grasp_T_uv_on_obj = xyz2uv(grasp_T_cam_on_obj,
        #                                    fx=camera['fx'],
        #                                    fy=camera['fy'],
        #                                    cx=camera['cx'],
        #                                    cy=camera['cy']
        #                                    )
        #         grasp_T_cam_on_obj_list.append(grasp_T_cam_on_obj)
        #         grasp_T_uv_on_obj_list.append(grasp_T_uv_on_obj)
        # grasp_T_uv_on_obj_list = np.array(grasp_T_uv_on_obj_list)
        # grasp_T_cam_on_obj_list =  np.vstack(grasp_T_cam_on_obj_list)
        ################################################################

        masks = []
        nocs_maps = []
        labels = []
        boxes_2d = []
        obj_names = []
        uid_mask = np.unique(mask)
        num_instance = 0
        obj_poses = []
        obj_scales = []

        for anno in annos:
            ###############  mask ##############
            uid = anno['uid']
            if uid not in uid_mask:
                # 有的物体在scene中被完全挡住了, mask中就不存在对应的uid了,
                # 但是anno中仍然记录了该被挡物体的信息, 需要筛出该物体
                continue
            mask_obj = mask == uid  # 单个物体的mask
            if mask_obj.flatten().tolist().count(True) < 100:
                # mask 面积过小也跳过
                continue
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
            xyz_obj = xyz.copy()
            scale = anno['scale']
            H_world = anno['pose']
            H_cam = extr.dot(H_world)
            H_inv = np.linalg.inv(H_cam)
            R = H_inv[:3, :3]
            T = H_inv[:3, 3]

            xyz_obj = (R @ xyz_obj.reshape(-1, 3).T).T + T  # 将物体的pose还原到原点
            xyz_obj = xyz_obj / scale  # 将物体还原为原始大小
            # xyz_obj[~mask_obj.flatten()] = 0 # 单个物体的partical点云, 非该物体的xyz置0
            nocs_para_obj = nocs_para[anno['obj_name']]  # 物体的 nocs 参数
            norm_factor, norm_corner = nocs_para_obj['norm_factor'], np.array(nocs_para_obj['norm_corner'])
            nocs_map = (xyz_obj - norm_corner[0]) * norm_factor + np.array((0.5, 0.5, 0.5)).reshape(1, 3) - 0.5 * (
                        norm_corner[1] - norm_corner[0]) * norm_factor
            nocs_map[~mask_obj.flatten()] = 0
            nocs_maps.append(nocs_map)
            ################ labels ###########
            labels.append(int(CLASSES[anno['obj_name'].split('/')[0]]))
            num_instance += 1
            obj_names.append(anno['obj_name'])
            obj_poses.append(H_cam)
            obj_scales.append(scale)

        nocs_maps = np.stack(nocs_maps, axis=-2).reshape(h, w, num_instance, 3)
        masks = np.stack(masks, axis=-1)
        boxes_2d = np.int32(boxes_2d)
        labels = np.int32(labels)
        obj_poses = np.float32(obj_poses)
        obj_scales = np.float32(obj_scales)
        np.savez_compressed(file=os.path.join(save_dir, i),
                            scene_id=scene_id,
                            obj_names=obj_names,
                            gt_masks=masks,  # shape (480, 640, num_instance)
                            gt_coords=nocs_maps,  # shape (480, 640, num_instance, 3)
                            img=rgb,
                            xyz = xyz,
                            extrinsic=extr,
                            gt_labels=labels,
                            gt_bboxes=boxes_2d,
                            gt_obj_poses=obj_poses,
                            gt_obj_scales=obj_scales,
                            # gripper
                            gt_gripper_quat =  grasp_quat_cam,
                            gt_gripper_T_uv = grasp_T_uv,
                            # gt_gripper_T_uv_on_obj = grasp_T_uv_on_obj_list,
                            # gt_gripper_T_cam_on_obj =  grasp_T_cam_on_obj_list,
                            gt_gripper_T_cam = grasp_T_cam,
                            gt_gripper_qual = grasp_qual,
                            gt_gripper_width =  grasp_width,
                            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument('--num-proc', type=int, default=1)
    args = parser.parse_args()
    preprocess_images(data_root=args.data_root)


    # from joblib import Parallel, delayed
    # Parallel(n_jobs=args.num_proc)(delayed(preprocess_images)(args.data_root) for i in range(args.num_proc))





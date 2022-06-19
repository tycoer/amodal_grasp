import os
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.spatial.transform import Rotation
import vedo
import tqdm
import json
import os
import h5py


def process_grasp_by_scene(csv_path):
    csv = pd.read_csv(csv_path)
    scene_id_unique = csv['scene_id'].unique()
    grasp_infos = {i: [] for i in scene_id_unique}
    for i in tqdm.tqdm(range(csv.__len__())):
        grasp_info = csv.iloc[i].to_numpy()
        grasp_infos[str(grasp_info[0])].append(np.float32(grasp_info[1:]).tolist())
    return grasp_infos

def get_extr(extrinsic):
    Q = extrinsic[0, :4]
    T = extrinsic[0, 4:]
    extr = np.eye(4)
    extr[:3, :3] = Rotation.from_quat(Q).as_matrix()
    extr[:3, 3] = T
    return extr

def pc_world_to_cam(pc_world, extr):
    pc_cam = (extr[:3, :3] @ pc_world.T).T + extr[:3, 3]
    return pc_cam

def depth2pc(depth, fx, fy, cx, cy, w, h, depth_scale=1, ):
    h_grid, w_grid= np.mgrid[0: h, 0: w]
    z = depth / depth_scale
    x = (w_grid - cx) * z / fx
    y = (h_grid - cy) * z / fy
    xyz = np.dstack((x, y, z))
    return xyz

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def xyz2uv(xyz, fx, fy, cx, cy):
    xyz=np.array(xyz).flatten()
    x,y=xyz[0]/xyz[2],xyz[1]/xyz[2]
    u,v=x*fx+cx,y*fy+cy
    uv=np.int0([u,v])
    return uv

def get_grasp_H(grasp_quat, grasp_T):
    H = np.eye(4)[np.newaxis].repeat(len(grasp_T), 0)
    R = Rotation.from_quat(grasp_quat).as_matrix()
    H[:, :3, :3] = R
    H[:, :3, 3] = grasp_T
    return H



class Open3dRender:
    def __init__(self, fx, fy, cx, cy):
        # 如果使用ssh服务器之类的, 请使用 X11 典型如软件 MobaXterm
        # 否则 self.vis.create_window(visible=False, width=self.w, height=self.h) 会失效导致后续无法rerender
        self.w, self.h = int(cx * 2), int(cy * 2)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False, width=self.w, height=self.h)
        self.view_opt = self.vis.get_view_control()

        # camera setting
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.intr = o3d.camera.PinholeCameraIntrinsic(fx=self.fx,
                                                     fy=self.fy,
                                                     cx=self.cx - 0.5,
                                                     cy=self.cy - 0.5,
                                                     width=self.w,
                                                     height=self.h)
        self.cam_para = o3d.camera.PinholeCameraParameters()
        self.cam_para.intrinsic = self.intr

    def run(self, mesh, extr):
        # 把图片变成正方形(非), 并且缩小分辨率(加快网络训练速度)
        extr = self.adjust_extr(extr)
        self.cam_para.extrinsic = extr
        self.vis.add_geometry(mesh)
        self.view_opt: o3d.visualization.ViewControl
        self.view_opt.convert_from_pinhole_camera_parameters(self.cam_para)

        color = np.array(self.vis.capture_screen_float_buffer(True))
        depth = np.array(self.vis.capture_depth_float_buffer(True))
        self.vis.remove_geometry(mesh)
        return color, depth, extr

    def adjust_extr(self, extr):
        extr_adjust = np.eye(4)
        extr_adjust[:3, :3] = Rotation.from_euler('xyz', (20, 0, 0), degrees=True).as_matrix() # 延x轴转30度
        extr_adjust = extr.dot(extr_adjust)
        return extr_adjust

def project_grasp_on_obj(scene: trimesh.Trimesh, grasp_H: np.ndarray, distance=1):
    '''

    :param scene: 整个场景的mesh (包括plane, object)
    :param grasp_H_world: gripper 的 transformation H (n, 4, 4)
    :param d:
    :return:
    '''
    # gripper的 坐标xyz位于 scene表面之外, 由于本算法基于2d图片, 所以需要将 gripper xyz坐标投射制scene表面之上(旋转 rot_xyz不变)
    # gripper夹爪的朝向是z轴, 故从 gripper的坐标xyz延z轴(gripper的z轴而不是世界坐标系的z轴)方向发射一条射线(射线的长度为distance, 由于intersectWithLine的输入参数为两个点所以需要指定一个distance算出距离gripper xyz distance且在gripper Z轴方向上 的点xyz
    # 射线与scene的交点既是位于scene表面的gripper xyz (既所求)
    num_grasp = len(grasp_H)
    scene_vedo = vedo.Mesh([np.array(scene.vertices), np.array(scene.faces)])
    grasp_R = grasp_H[:, :3, :3] # (num_grasp , 3, 3)
    grasp_T = grasp_H[:, :3, 3] # (nu_grasp, 3)
    T_relative = np.array([[0, 0, distance]]).T # (3, 1)
    T_relative = np.array([R @ T_relative for R in grasp_R]).reshape(num_grasp, 3) # (num_grasp, 3)
    # vedo中计算交点的api (效果很好): intersectWithLine, [0] 的原因 intersectWithLine 会计算多个交点, [0]是选取射线第一次击中scene的点
    ray_start = grasp_T
    ray_end = grasp_T + T_relative

    T_on_obj = []
    project_success = []
    for start, end in zip(ray_start, ray_end):
        T = scene_vedo.intersectWithLine(start.tolist(), end.tolist())
        if len(T) != 0:
            project_success.append(1)
            T_on_obj.append(T[0])
        else:
            project_success.append(0)
            T_on_obj.append(start)
    T_on_obj = np.array(T_on_obj)
    project_success = np.array(project_success)
    # scene.ray.intersects_location() 这个是trimesh中的计算交点的api, 但经测试效果奇差无比, 具体体现为计算出交点的位置不对(越是垂直于射线世界坐标系下的xyz轴结果越准, 反之结果越不准)
    grasp_H_on_obj = grasp_H.copy()
    grasp_H_on_obj[:, :3, 3] = T_on_obj
    return grasp_H_on_obj, project_success

if __name__ == '__main__':
    data_root = '/disk1/data/giga/data_packed_train_raw'
    rerender = True
    debug = False



    grasp_path = os.path.join(data_root, 'grasps.csv')
    mesh_pose_list_root = os.path.join(data_root, 'mesh_pose_list')
    scene_root = os.path.join(data_root, 'scenes')
    setup_path = os.path.join(data_root, 'setup.json')
    grasp_by_scene_path = os.path.join(data_root, 'grasps_by_scene.h5')
    scene_cam_root = os.path.join(data_root, 'scenes_cam')
    urdf_root = os.path.join(data_root, 'urdfs')
    plane_path = os.path.join(urdf_root, 'setup/plane.obj')

    graps_cam_by_scene_path = os.path.join(data_root, 'grasps_cam_by_scene.h5')
    grasp_cam_path = os.path.join(data_root, 'grasps_cam.csv')
    setup_rerender_path = os.path.join(data_root, 'setup_rerender.json')

    os.makedirs(scene_cam_root, exist_ok=True)


    ################## 相机内参设置 ##############
    with open(setup_path, 'r') as f :
        setup = json.load(f)
    K = setup['intrinsic']['K']
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    # init open3d render
    if rerender == True:
        fx, fy, cx, cy = 1100, 1100, 240, 240
        render = Open3dRender(fx=fx, fy=fy, cx=cx, cy=cy)
        setup_rerender = setup.copy()
        setup_rerender['intrinsic'] = dict(width=int(cx * 2),
                                           height=int(cy * 2),
                                           K=[fx, 0, cx, 0, fy, cy, 0, 0, 1])
        with open(setup_rerender_path, 'w') as f:
            json.dump(setup_rerender, f)
    w, h = int(cx * 2), int(cy * 2)
    ################### 文件初始化 ##############
    # 以scene为基础合并grasp
    if os.path.exists(grasp_by_scene_path) == True:
        print(f'{grasp_by_scene_path} detected, using the file.')
        grasp_by_scene_h5 = h5py.File(grasp_by_scene_path, 'r')

    else:
        print(f'{grasp_by_scene_path} is not detected, processing grasp...')
        grasp_by_scene = process_grasp_by_scene(grasp_path)
        grasp_by_scene_h5 = h5py.File(grasp_by_scene_path, mode='w')
        for k, v in grasp_by_scene.items():
            grasp_by_scene_h5[k] = v
        grasp_by_scene_h5.close()
    print(f'{len(grasp_by_scene_h5)} scene detected, processing...')

    # init csv and h5
    assert os.path.exists(graps_cam_by_scene_path) == False, f'{graps_cam_by_scene_path} exists! Remove the file first!'
    assert os.path.exists(grasp_cam_path) == False,  f'{grasp_cam_path} exists! Remove the file first!'
    grasp_cam_by_scene_h5 = h5py.File(graps_cam_by_scene_path, mode='w')
    columns = ['scene_id', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z', 'width', 'label', 'u', 'v', 'u_obj', 'v_obj', 'x_obj', 'y_obj', 'z_obj', 'project_success', 'project_success_uv']
    grasp_cam = pd.DataFrame(columns=columns)
    grasp_cam.to_csv(grasp_cam_path, mode='w', index=False)
    ###############################################################


    for scene_id, grasp_world_data in tqdm.tqdm(grasp_by_scene_h5.items()):
        ################## world 坐标系下的处理 ######################
        grasp_world_data = np.array(grasp_world_data)
        grasp_quat_world = grasp_world_data[:, :4]
        grasp_T_world = grasp_world_data[:, 4:7]
        grasp_H_world = get_grasp_H(grasp_quat_world, grasp_T_world)


        ann_path = os.path.join(mesh_pose_list_root, scene_id + '.npz')
        scene_path = os.path.join(scene_root, scene_id + '.npz')
        scene = dict(np.load(scene_path, allow_pickle=True))
        ann = dict(np.load(ann_path, allow_pickle=True))['pc']
        # 还原场景
        scene_trimesh = trimesh.Scene()
        # 在场景中放置平面
        plane = trimesh.load(plane_path)
        plane: trimesh.Trimesh
        plane.apply_transform(np.array([[1, 0, 0, 0.15],
                                        [0, 1, 0, 0.15],
                                        [0, 0, 1, 0.051],
                                        [0, 0, 0, 1]]))
        scene_trimesh.add_geometry(plane)
        # 在场景中放置物体
        for obj_name, scale, obj_pose_world in ann:
            obj_path = os.path.join(urdf_root, obj_name[11:])
            mesh = trimesh.load(obj_path)
            mesh: trimesh.Trimesh
            mesh.apply_scale(scale)
            mesh.apply_transform(obj_pose_world)
            scene_trimesh.add_geometry(mesh)
        scene_trimesh = as_mesh(scene_trimesh)
        # grasp 的坐标投射到物体上
        grasp_H_world_on_obj, project_success = project_grasp_on_obj(scene_trimesh, grasp_H_world)
        project_success = project_success.reshape(-1, 1)
        #################### camera 坐标系下的处理 ##################
        depth, extr = scene['depth_imgs'][0], scene['extrinsics']
        extr = get_extr(extr)
        if rerender == True:
            scene_o3d = scene_trimesh.as_open3d
            _, depth, extr = render.run(scene_o3d, extr)
        pc_cam = depth2pc(depth, fx, fy, cx, cy, w, h)


        grasp_H_cam = np.array([extr.dot(H) for H in grasp_H_world])
        grasp_H_cam_on_obj = np.array([extr.dot(H) for H in grasp_H_world_on_obj])
        grasp_T_cam = grasp_H_cam[:, :3, 3]
        grasp_T_cam_on_obj = grasp_H_cam_on_obj[:, :3, 3]
        grasp_quat_cam = Rotation.from_matrix(grasp_H_cam[:, :3, :3]).as_quat()
        grasp_uv_cam_on_obj =  np.array([xyz2uv(T, fx, fy, cx, cy) for T in grasp_T_cam_on_obj])
        grasp_uv_cam = np.array([xyz2uv(T, fx, fy, cx, cy) for T in grasp_T_cam])

        uv_valid = np.hstack((grasp_uv_cam_on_obj, grasp_uv_cam))
        project_success_uv = np.logical_not((uv_valid > w).any(axis=1) | (uv_valid < 0).any(axis=1)).reshape(-1, 1)

        #################### 保存 #################################

        grasp_cam_data = grasp_world_data.copy()
        grasp_cam_data[:, :4] = grasp_quat_cam
        grasp_cam_data[:, 4:7] = grasp_T_cam
        grasp_cam_data = np.hstack((grasp_cam_data, grasp_uv_cam, grasp_uv_cam_on_obj, grasp_T_cam_on_obj, project_success, project_success_uv))
        if debug == True:
            pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_cam.reshape(-1, 3)))
            axis_list = []
            for H in grasp_H_cam[:4]:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
                axis.transform(H)
                axis_list.append(axis)
            box_list = []
            for H in grasp_H_cam_on_obj[:4]:
                box = o3d.geometry.TriangleMesh.create_box(0.005, 0.005, 0.005)
                box: o3d.geometry.TriangleMesh
                box.paint_uniform_color([0, 1, 1])
                box.transform(H)
                box_list.append(box)
            o3d.visualization.draw_geometries([pc_o3d, scene_o3d.transform(extr)] + axis_list + box_list)
        grasp_cam_by_scene_h5[scene_id] = grasp_cam_data.tolist()
        grasp_cam = pd.DataFrame(data=np.hstack((np.array([scene_id] * len(grasp_cam_data))[:, np.newaxis], grasp_cam_data)))
        np.savez_compressed(os.path.join(scene_cam_root, f'{scene_id}.npz'),
                 extrinsic=extr,
                 depth=depth,
                 pc=pc_cam)
        grasp_cam.to_csv(grasp_cam_path, mode='a+', index=False, header=False)
    grasp_by_scene_h5.close()
    grasp_cam_by_scene_h5.close()
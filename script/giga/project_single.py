import os
import numpy as np
import open3d as o3d
import json
from scipy.spatial.transform import Rotation
import trimesh

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
        extr_adjust[:3, :3] = Rotation.from_euler('xyz', (30, 0, 0), degrees=True).as_matrix() # 延x轴转30度
        extr_adjust = extr.dot(extr_adjust)
        return extr_adjust

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

def get_extr(extrinsic):
    Q = extrinsic[0, :4]
    T = extrinsic[0, 4:]
    extr = np.eye(4)
    extr[:3, :3] = Rotation.from_quat(Q).as_matrix()
    extr[:3, 3] = T
    return extr

def depth2pc(depth, fx, fy, cx, cy, w, h, depth_scale=1, ):
    h_grid, w_grid= np.mgrid[0: h, 0: w]
    z = depth / depth_scale
    x = (w_grid - cx) * z / fx
    y = (h_grid - cy) * z / fy
    xyz = np.dstack((x, y, z))
    return xyz


if __name__ == '__main__':
    scene_id = '2562aaccd0824ec186c6c4583410de10'
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

    ann_path = os.path.join(mesh_pose_list_root, f'{scene_id}.npz')
    ann = np.load(ann_path, allow_pickle=True)['pc']
    scene_path = os.path.join(scene_root, scene_id + '.npz')
    scene = dict(np.load(scene_path, allow_pickle=True))

    with open(setup_path, 'r') as f :
        setup = json.load(f)
    K = setup['intrinsic']['K']
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    # init open3d render
    if rerender == True:
        fx, fy, cx, cy = 350, 350, 128, 128
        render = Open3dRender(fx=fx, fy=fy, cx=cx, cy=cy)
        setup_rerender = setup.copy()
        setup_rerender['intrinsic'] = dict(width=int(cx * 2),
                                           height=int(cy * 2),
                                           K=[fx, 0, cx, 0, fy, cy, 0, 0, 1])
        with open(setup_rerender_path, 'w') as f:
            json.dump(setup_rerender, f)
    w, h = int(cx * 2), int(cy * 2)

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

    depth, extr = scene['depth_imgs'][0], scene['extrinsics']
    extr = get_extr(extr)

    if rerender == True:
        scene_o3d = scene_trimesh.as_open3d
        _, depth, extr = render.run(scene_o3d, extr)
    pc_cam = depth2pc(depth, fx, fy, cx, cy, w, h)
    np.savez_compressed(os.path.join(scene_cam_root, f'{scene_id}.npz'),
                        extrinsic=extr,
                        pc=pc_cam)
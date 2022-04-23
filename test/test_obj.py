import open3d as o3d
import trimesh
import pybullet as p
if __name__ =='__main__':
    obj_path = '/disk3/data/amodal_grasp/obj_models_new/cell_phone/d7ed512f7a7daf63772afc88105fa679/model.obj'
    # mesh = trimesh.load(obj_path, force='mesh')
    p.connect(p.GUI)  # 必须先连接服务器
    p.setGravity(0, 0, -9.8)
    mesh = p.loadURDF('/disk3/data/amodal_grasp/obj_models_new/cell_phone/d7ed512f7a7daf63772afc88105fa679/model.urdf')


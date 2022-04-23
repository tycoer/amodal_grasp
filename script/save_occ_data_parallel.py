import os
import glob
import time
import argparse
import numpy as np
import multiprocessing as mp
import trimesh
from utils.implicit import URDF, sample_iou_points, as_mesh

def get_scene_from_mesh_pose_list(mesh_pose_list, scene_as_mesh=True, return_list=False):
    # create scene from meshes
    scene = trimesh.Scene()
    mesh_list = []
    # for mesh_path, scale, pose in mesh_pose_list:#tycoer
    for mesh_pose in mesh_pose_list:#tycoer
        mesh_path, scale, pose = mesh_pose['mesh_path'], mesh_pose['scale'], mesh_pose['pose']
        mesh_path = '/disk1/data/amodal_grip_dataset/'+mesh_path[16:]


        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1
            assert len(obj.links[0].visuals) == 1
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)
            # if isinstance(mesh, trimesh.Scene):
            #     mesh = as_mesh(mesh)
        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        scene.add_geometry(mesh)
        mesh_list.append(mesh)
    if scene_as_mesh:
        scene = as_mesh(scene)
    if return_list:
        return scene, mesh_list
    else:
        return scene



def sample_occ(mesh_pose_list_path, num_point, uniform):
    mesh_pose_list = list(np.load(mesh_pose_list_path, allow_pickle=True))
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True)
    points, occ = sample_iou_points(mesh_list, scene.bounds, num_point, uniform=uniform)
    return points, occ

def save_occ(mesh_pose_list_path, args):
    points, occ = sample_occ(mesh_pose_list_path, args.num_point_per_file * args.num_file, args.uniform)
    points = points.astype(np.float16).reshape(args.num_file, args.num_point_per_file, 3)
    occ = occ.reshape(args.num_file, args.num_point_per_file)
    name = os.path.basename(mesh_pose_list_path)[:-4]
    save_root = os.path.join(args.raw, 'occ', name)
    os.makedirs(save_root)
    for i in range(args.num_file):
        np.savez(os.path.join(save_root, '%04d.npz' % i), points=points[i], occ=occ[i])

def log_result(result):
    g_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    mesh_list_files = glob.glob(f'{args.raw}/mesh_pose_list/*.npy')
    

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time

    g_num_total_jobs = len(mesh_list_files)
    g_completed_jobs = []

    g_starting_time = time.time()

    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        for f in mesh_list_files:
            pool.apply_async(func=save_occ, args=(f,args), callback=log_result)
        pool.close()
        pool.join()
    else:
        for f in mesh_list_files:
            save_occ(f, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("raw", type=str)
    parser.add_argument("num_point_per_file", type=int)
    parser.add_argument("num_file", type=int)
    parser.add_argument("--uniform", action='store_true', help='sample uniformly in the bbox, else sample in the tight bbox')
    args = parser.parse_args()
    main(args)

    # from joblib import Parallel, delayed
    # Parallel(n_jobs=args.n um_proc)(delayed(main)(args) for _ in range(args.num_proc))


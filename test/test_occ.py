import numpy as np
import open3d as o3d
occ_path = '/hdd0/data/giga/data_packed_train_raw/occ/000a51d1b5cd4ca5aeb6ee3d0b24d285/0000.npz'
occ = dict(np.load(occ_path))
occ_points = occ['points']
occ_value = occ['occ']

occ_points = occ_points[occ_value]

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(occ_points))

o3d.io.write_point_cloud('/home/hanyang/occ.ply', pc)
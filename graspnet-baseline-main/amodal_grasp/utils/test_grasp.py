import open3d as o3d
import numpy as np
anno = dict(np.load('/mnt/g/grasp_label/000_labels.npz'))
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(anno['points']))
o3d.visualization.draw_geometries([pc])
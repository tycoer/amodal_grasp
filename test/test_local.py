import open3d as o3d
import numpy as np

# import cv2

# cv2.imshow('afdafds', np.zeros((3, 100, 100),dtype='uint8'))
a = np.random.rand(100, 3)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a))
o3d.visualization.draw_geometries([pc])
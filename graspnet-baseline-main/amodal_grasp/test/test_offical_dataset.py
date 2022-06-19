__author__ = 'mhgou'
__version__ = '1.0'

from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import cv2
import numpy as np

# GraspNetAPI example for loading grasp for a scene.
# change the graspnet_root path

####################################################################
graspnet_root = '/disk5/data/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

sceneId = 1
annId = 3

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
print('6d grasp:\n{}'.format(_6d_grasp))

# _6d_grasp is an GraspGroup instance defined in grasp.py
print('_6d_grasp:\n{}'.format(_6d_grasp))
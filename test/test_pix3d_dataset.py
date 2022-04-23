from model_2d.pix3d_dataset import Pix3DDataset
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import torch
from model_2d.utils.shape import box2D_to_cuboid3D, cuboid3D_to_unitbox3D


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPix3D', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MinMaxNormalize'),
    dict(type='Pad', size_divisor=32),
    dict(type='Pix3DPipeline'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_meshes', 'gt_voxels', 'gt_masks', 'Ks', 'gt_zs'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'item', 'iscrowd')),
]



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MinMaxNormalize'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'item')
                 ),
        ])
]


def test_cubify(data):
    vox = data['gt_voxels'].data



def vis_after_pipeline(data):
    img = data['img'].data.cpu().numpy()
    voxels = data['gt_voxels'].data[0].cpu().numpy()
    K = data['Ks'].data[0].cpu().numpy()
    verts, faces = data['gt_meshes'].data
    mask = data['gt_masks'].data.masks[0]
    boxes = data['gt_bboxes'].data.cpu().numpy()

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame() # 默认在原点 (0, 0, 0)
    voxels_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voxels))
    # o3d.io.write_point_cloud('/home/hanyang/voxel.ply', voxels_o3d)
    mesh_o3d = o3d.geometry.TriangleMesh()
    verts = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img.shape[2], height=img.shape[1])
    vis.add_geometry(voxels_o3d)
    # vis.add_geometry(axis)
    vis.add_geometry(mesh_o3d)
    render_opt = vis.get_render_option()
    view_opt = vis.get_view_control()
    intr = o3d.camera.PinholeCameraIntrinsic(fx=K[0],
                                             fy=K[0],
                                             cx=K[1] - 0.5,
                                             cy=K[2] - 0.5,
                                             width=img.shape[2],
                                             height=img.shape[1])
    extr = np.eye(4)
    cam_para = o3d.camera.PinholeCameraParameters()
    cam_para.extrinsic = extr
    cam_para.intrinsic = intr
    view_opt: o3d.visualization.ViewControl
    view_opt.convert_from_pinhole_camera_parameters(cam_para)

    img_out = vis.capture_screen_float_buffer(True)
    img_with_bbox = np.ascontiguousarray(img.copy().transpose(1, 2, 0)[:, :, ::-1], dtype=np.float32)
    for i in np.int32(boxes):
        cv2.rectangle(img_with_bbox, i[:2], i[2:], (0, 255, 0), 5)

    # o3d.visualization.draw_geometries([voxels_o3d, axis])
    img_show = np.hstack((img_with_bbox, img_out,np.dstack([mask, mask, mask])))
    plt.imshow(img_show)
    plt.show()
    return img_show


if __name__ == '__main__':

    dataset_train = Pix3DDataset(data_root='/disk2/data/pix3d',
                           anno_path='/disk2/data/pix3d/pix3d_s1_train.json',
                           pipeline=train_pipeline)
    data = dataset_train[5683]
    # img_show = vis_after_pipeline(data)

    # voxel = data['gt_voxels'].data[0].cpu().numpy()
    # pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voxel))
    # o3d.io.write_point_cloud('/home/hanyang/vox.ply', pc)
    #
    # img = data['img'].data.cpu().numpy()
    # img = np.ascontiguousarray(img.copy().transpose(1, 2, 0)[:, :, ::-1], dtype=np.float32)
    # plt.imshow(img)
    # plt.show()























    # dataset_test = Pix3DDataset(data_root='/disk2/data/pix3d',
    #                             anno_path='/disk2/data/pix3d/pix3d_s1_test.json',
    #                             pipeline=test_pipeline)

    # data = dataset_test[0]
    # save_root = '/home/hanyang/img'
    # import os
    # os.makedirs(save_root, exist_ok=True)
    # for i in range(100):
    #     data = dataset[i]
    #     img_show = vis_after_pipeline(data)
    #     cv2.imwrite(f'{save_root}/i.jpg', img_show)



    # for i in range(dataset.__len__()):
    #     dataset[i]
    # vis_after_pipeline(data)
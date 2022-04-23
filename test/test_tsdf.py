import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.Gray32,
        )

    def integrate(self, depth_img, intrinsic, extrinsic, mask=None):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """

        # mask = normalization(mask)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)) if mask is None else o3d.geometry.Image(mask),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=100,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        # extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        voxels = self._volume.extract_voxel_grid().get_voxels()
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_grid[0, i, j, k] = voxel.color[0]
        return tsdf_grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()

class Camera:
    def __init__(self, fx, fy, cx, cy, height, width):
        self.fx = fx
        self.fy=fy
        self.cx = cx
        self.cy = cy
        self.height = height
        self.width = width

def from_list(list):
    H = np.eye(4)
    rotation = Rotation.from_quat(list[:4])
    translation = list[4:]

    H[:3, :3] =  rotation.as_matrix()
    H[:3, 3] = translation


    return H

if __name__ == '__main__':
    tsdf = TSDFVolume(0.3, 120)
    cam = Camera(540, 540, 320, 240, 480, 640)
    img_path = '/home/hanyang/amodal_grisp/data_mini/scenes/4357098be7d742a590f335090012b774.npz'
    imgs = dict(np.load(img_path))
    for depth, extr, mask in zip(imgs['depth_imgs'], imgs['extrinsics'], imgs['mask_imgs']):
        tsdf.integrate(np.float32(mask / 10), cam, from_list(extr))

    o3d.io.write_point_cloud('/home/hanyang/tsdf.ply',tsdf.get_cloud())

    pc = tsdf.get_cloud()
    color = np.int8(pc.colors)
    mask = color[:, 0]
    points = np.array(pc.points)

    pcs = []
    for i in np.unique(mask):
        pcs.append(points[mask==i])
        p  =  o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[mask==i]))
        o3d.io.write_point_cloud(f'/home/hanyang/mask{i}.ply', p)

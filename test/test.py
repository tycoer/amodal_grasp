
import numpy as np
import cv2

def handeye_affineCalculation(cam_points, rob_points, eye_in_hand=True, th_1=0, th_2=0):
    '''
    计算仿射矩阵, 用于9点标定

    注意: 无论眼在手外,还是眼在手上, 相机都要垂直与标定板所在平面, 且机械臂移动过程中不要出现旋转(即rx,ry,rz的变化)
          若出现 rx, ry, rz的变化, 9点标定便不再适用, 请搜索 cv2.calibrateHandeye 的使用
    eye_in_hand 下 (即相机安装在机械臂末端随机械臂移动,标定板固定在某处 -- 移动的相机拍固定的标定板), 
    首先应保证机械臂移动点相对于一个中心点对称,即机械臂应按s型路线走"田", 如此便能保证所有机械臂点关于'田'中心点对称
    这是因为由于相机的镜像关系,如果将机械臂点与相机点顺序成对(即rob_point1,cam_point1|rob_point2,cam_point2|...)送入公式进行仿射矩阵的计算, 如将新点带入会使机械臂向新点的镜像位置行进
    该镜像错误有如下解决办法:
    1.将机械臂点与相机点倒序成对带入(即rob_point1,cam_point9|rob_point2,cam_point8|...),直接可求出正确的仿射矩阵
    2.将机械臂点与相机点顺序成对带入(即rob_point1,cam_point1|rob_point2,cam_point2|...), 但需对仿射矩阵作如下处理:
      R-旋转部分四个元素取负(即对affine_matrix[:,:2]取负,取负的几何意义为将矩阵旋转180度)
      T-平移部分tx'=2*cx-tx,ty'=2*cy-ty, 其中(cx,cy)为机械臂采样点里的中心点坐标, tx,ty分别为顺序成对代入求出的仿射矩阵的affine_matrix[0,2],affine_matrix[1,2]
      R:affine_matrix[:,:2]=-affine_matrix[:,:2]
      T:affine_matrix[0,2]=2*cx-affine_matrix[0,2],affine_matrix[1,2]=2*cy-affine_matrix[0,2]
    此外, 每次工作,机械臂必须回到中心点拍照

    eye_to_hand(即相机固定架子上, 标定板固定在机械臂末端随机械臂末端移动 -- 固定的相机拍移动的标定板), 
    唯一的要求为:采样阶段应尽量保持机械臂末端贴近工作平面, 并保证同一平面上取9个点(位置任意,不需保证对称),工作时也无需回到某个固定点

    Parameters
    ----------
    cam_points : array
        DESCRIPTION. 相机对标定板采集的像素点坐标, (nx2) 的矩阵
    rob_points : array
        DESCRIPTION. 相机每次拍照所对应的机械臂姿态(对于机械臂姿态(x,y,z,rx,ry,rz)只保留(x,y)), (nx2)的矩阵
    eye_in_hand : bool, optional
        DESCRIPTION. The default is True. True为眼在手上, False 为眼在手外 
    th_1 : float, optional
        DESCRIPTION. The default is 0.
    th_2 : float, optional
        DESCRIPTION. The default is 0.

    关于th_1, 及th_2 的说明
    为什么要有这两个参数?
    eye_in_hand 下:由于此次模式下必须保证'田' 中心点与标定板的中心点的对齐((x,y)对齐即可, 一般这个对齐过程为
    开启机械臂的自由移动模式, 然后手动将机械臂末端拉到标定板中心点上方), 但是由于某些原因导致相机拍不到标定板(例如机械臂放得比较低,
    1,2,3点可以拍得到, 但4,5点拍不到), 这样就需要将机械臂末端移动到某个合适位置以保证一次性9个点都能拍到, 这样'某个合适位置' 就与标定板中心点
    有偏离, 所以 th 用于描述这个偏离
    例如 '某个合适位置(机器人末端中心点, 有可能是xy, xz, yz, 具体看你标定哪一个平面)' 坐标为(1.5,3.5)
        标定板中心点坐标为(2, 2.7)
        故 th_1 = 2 - 1.5 = 1
           th_2 = 2.7 - 3.5 = -0.8

    eye_to_hand 下: 由于标定板安装在机械臂末端, 故标定板中心点与'田'仍存在偏差
    Returns
    -------
    affine_matrix : array
        DESCRIPTION.仿射矩阵(2x3)

    '''

    cam_points = np.float32(cam_points)
    rob_points = np.float32(rob_points)
    if len(cam_points) == len(rob_points) and (len(cam_points) and len(rob_points) >= 3):
        if eye_in_hand == True:
            cam_points_inverse = cam_points[::-1, :]
            affine_matrix = cv2.estimateAffine2D(cam_points_inverse, rob_points)[0]
        else:
            affine_matrix = cv2.estimateAffine2D(cam_points, rob_points)[0]
        affine_matrix[0, 2] += th_1
        affine_matrix[1, 2] += th_2
    else:
        affine_matrix = None
    if affine_matrix is not None:
        affine_matrix[0, 2] += th_1
        affine_matrix[1, 2] += th_2
    return affine_matrix


def handeye_affineTransform(affine_matrix, cam_point):
    '''
    仿射变换, 输入标定文件与一个像素点坐标, 求解机器人坐标

    Parameters
    ----------
    affine_matrix : array 
        DESCRIPTION. 仿射矩阵 (2x3)
    cam_point : array
        DESCRIPTION. 像素点坐标(1x2)

    Returns
    -------
    rob_point : array
        DESCRIPTION. 机器人坐标(1x2)

    '''

    cam_point, affine_matrix = np.float32(cam_point).reshape(-1, 2), np.float32(affine_matrix)
    rob_point = np.vstack(
        (affine_matrix[0, 0] * cam_point[:, 0] + affine_matrix[0, 1] * cam_point[:, 1] + affine_matrix[0, 2],
         affine_matrix[1, 0] * cam_point[:, 0] + affine_matrix[1, 1] * cam_point[:, 1] + affine_matrix[1, 2])).T
    return rob_point


if __name__ == '__main__':
    # 造数据
    # 机器人末端xy的坐标值 一般单位为米
    rob_points_xy = np.array([[1, 1],
                               [0, 1],
                               [-1, 1],
                               [-1, 0],
                               [0, 0],
                               [1, 0],
                                  [1, -1],
                                  [0, -1],
                                  [-1, -1]])

    # 图片中标志物的坐标位置单位为像素, 这里为了方便, 直接 +50 代表像素值
    # 标志物一般采用 aruco
    cam_points = rob_points_xy + 50

    affine = handeye_affineCalculation(cam_points=cam_points,
                                       rob_points=rob_points_xy,
                                       eye_in_hand=True,
                                       th_1=0.1,
                                       th_2=0.1)
    # 验证
    rob_point_eval = handeye_affineTransform(affine_matrix=affine,
                            cam_point=cam_points[0])
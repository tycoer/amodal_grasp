import numpy as np
import matplotlib.pyplot as plt

def getRANSACInliers(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=1):
    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = np.arange(SourceHom.shape[1])
    for i in range(0, MaxIterations):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(SourceHom.shape[1], size=5)
        _, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        if Residual < BestResidual:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break

        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('Inlier ratio: ', InlierRatio)

    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio


def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    # Auto-parameter selection based on source-target heuristics
    TargetNorm = np.mean(np.linalg.norm(target, axis=1))
    SourceNorm = np.mean(np.linalg.norm(source, axis=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST if(RatioST>RatioTS) else RatioTS
    StopT = PassT / 100
    nIter = 100
    if verbose:
        print('Pass threshold: ', PassT)
        print('Stop threshold: ', StopT)
        print('Number of iterations: ', nIter)

    SourceInliersHom, TargetInliersHom, BestInlierRatio = getRANSACInliers(SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT)

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scales:', Scales)

    return Scales, Rotation, Translation, OutTransform


def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs
# def align(bbox_results, segm_results, nocs_results, depth, labels, img_meta,
#           save_path=None, if_norm=False, with_scale=True, verbose=False):
#     if type(depth) != np.ndarray:
#         depth = depth.cpu().numpy()[0]
#     class_ids = img_meta[0]['cat_ids']
#     # class_ids = [class_ids[i] for i in labels]
#     intrinsics = np.array(img_meta[0]['intrinsics'])
#     num_instances = sum([bbox_results[i].shape[0] for i in range(len(class_ids))])
#     error_messages = ''
#     elapses = []
#     if num_instances == 0:
#         return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages, elapses
#
#     cls_RTs = [[] for _ in range(len(class_ids))]
#     cls_scales = [[] for _ in range(len(class_ids))]
#     # RTs = np.zeros((num_instances, 4, 4))
#     # bbox_scales = np.ones((num_instances, 3))
#
#     for label in range(len(bbox_results)):
#         bbox_scales = []
#         RTs = []
#
#         bboxes = bbox_results[label]
#         segms = segm_results[label]
#         nocs = nocs_results[label]
#         for i in range(bboxes.shape[0]):
#             class_id = class_ids[label]
#             mask = segms[i]
#             if mask.sum() == 0:
#                 aligned_RT = np.identity(4, dtype=np.float32)
#
#                 # print('Estimation takes {:03f}s.'.format(time.time() - start))
#                 # from camera world to computer vision frame
#                 z_180_RT = np.zeros((4, 4), dtype=np.float32)
#                 z_180_RT[:3, :3] = np.diag([-1, -1, 1])
#                 z_180_RT[3, 3] = 1
#
#                 # RTs[i, :, :] = z_180_RT @ aligned_RT
#                 RT = z_180_RT @ aligned_RT
#
#                 bbox_scale = np.array([1., 1., 1.], dtype=np.float32)
#
#                 bbox_scales.append(bbox_scale)
#                 RTs.append(RT)
#                 continue
#             coord = nocs[i]
#
#             abs_coord_pts = np.abs(coord[mask == 1] - 0.5)
#             bbox_scale = 2 * np.amax(abs_coord_pts, axis=0)
#             # bbox_scales[i, :] = 2 * np.amax(abs_coord_pts, axis=0)
#
#             pts, idxs = backproject(depth, intrinsics, mask)
#             coord_pts = coord[idxs[0], idxs[1], :] - 0.5
#
#             if if_norm:
#                 # scale = np.linalg.norm(bbox_scales[i, :])
#                 scale = np.linalg.norm(bbox_scale[i, :])
#                 # bbox_scales[i, :] /= scale
#                 bbox_scale /= scale
#                 coord_pts /= scale
#
#             try:
#                 # start = time.time()
#
#                 scales, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
#
#                 aligned_RT = np.zeros((4, 4), dtype=np.float32)
#                 if with_scale:
#                     aligned_RT[:3, :3] = np.diag(scales) / 1000 @ rotation.transpose()
#                 else:
#                     aligned_RT[:3, :3] = rotation.transpose()
#                 aligned_RT[:3, 3] = translation / 1000
#                 aligned_RT[3, 3] = 1
#
#                 if save_path is not None:
#                     coord_pts_rotated = aligned_RT[:3, :3] @ coord_pts.transpose() + aligned_RT[:3, 3:]
#                     coord_pts_rotated = coord_pts_rotated.transpose()
#                     np.savetxt(save_path + '_{}_{}_depth_pts.txt'.format(i, class_id), pts)
#                     np.savetxt(save_path + '_{}_{}_coord_pts.txt'.format(i, class_id), coord_pts)
#                     np.savetxt(save_path + '_{}_{}_coord_pts_aligned.txt'.format(i, class_id), coord_pts_rotated)
#
#                 if verbose:
#                     print('Mask ID: ', i)
#                     print('Scale: ', scales / 1000)
#                     print('Rotation: ', rotation.transpose())
#                     print('Translation: ', translation / 1000)
#
#                 # elapsed = time.time() - start
#                 # print('elapsed: ', elapsed)
#                 # elapses.append(elapsed)
#
#
#             except Exception as e:
#                 message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(label + 1,
#                                                                                             img_meta[0]['img_path'].split('/')[-2:], str(e))
#                 print(message)
#                 error_messages += message + '\n'
#                 aligned_RT = np.identity(4, dtype=np.float32)
#
#                 # print('Estimation takes {:03f}s.'.format(time.time() - start))
#             # from camera world to computer vision frame
#             z_180_RT = np.zeros((4, 4), dtype=np.float32)
#             z_180_RT[:3, :3] = np.diag([-1, -1, 1])
#             z_180_RT[3, 3] = 1
#
#             # RTs[i, :, :] = z_180_RT @ aligned_RT
#             RT = z_180_RT @ aligned_RT
#
#             bbox_scales.append(bbox_scale)
#             RTs.append(RT)
#
#         cls_scales[label].append(bbox_scales)
#         cls_RTs[label].append(RTs)
#
#     return cls_RTs, cls_scales, error_messages, elapses

def align(bbox_results, segm_results, nocs_results, depth, det_labels, img_metas,
          save_path=None, if_norm=False, with_scale=True, verbose=False):
    if type(depth) != np.ndarray:
        depth = depth.cpu().numpy()[0]
    intrinsics = img_metas[0]['intrinsics']
    num_instances = len(det_labels)
    error_messages = ''
    elapses = []
    if num_instances == 0:
        return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages, elapses

    cls_RTs = [[] for _ in range(len(segm_results))]
    cls_scales = [[] for _ in range(len(segm_results))]
    # RTs = np.zeros((num_instances, 4, 4))
    # bbox_scales = np.ones((num_instances, 3))

    for i in range(len(bbox_results)):
        bbox_scales = []
        RTs = []
        bboxes = bbox_results[i]
        segms = segm_results[i]
        nocs = nocs_results[i]
        for j in range(bboxes.shape[0]):
            # class_id = det_labels[j]
            mask = segms[j]
            if mask.sum() == 0:
                aligned_RT = np.identity(4, dtype=np.float32)

                # print('Estimation takes {:03f}s.'.format(time.time() - start))
                # from camera world to computer vision frame
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1

                # RTs[i, :, :] = z_180_RT @ aligned_RT
                RT = z_180_RT @ aligned_RT

                bbox_scale = np.array([1., 1., 1.], dtype=np.float32)

                bbox_scales.append(bbox_scale)
                RTs.append(RT)
                continue
            coord = nocs[j]

            abs_coord_pts = np.abs(coord[mask == 1] - 0.5)
            bbox_scale = 2 * np.amax(abs_coord_pts, axis=0)
            # bbox_scales[i, :] = 2 * np.amax(abs_coord_pts, axis=0)

            pts, idxs = backproject(depth, intrinsics, mask)
            coord_pts = coord[idxs[0], idxs[1], :] - 0.5

            if if_norm:
                # scale = np.linalg.norm(bbox_scales[i, :])
                scale = np.linalg.norm(bbox_scale[i, :])
                # bbox_scales[i, :] /= scale
                bbox_scale /= scale
                coord_pts /= scale

            try:
                # start = time.time()

                scales, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)

                aligned_RT = np.zeros((4, 4), dtype=np.float32)
                if with_scale:
                    aligned_RT[:3, :3] = np.diag(scales) / 1000 @ rotation.transpose()
                else:
                    aligned_RT[:3, :3] = rotation.transpose()
                aligned_RT[:3, 3] = translation / 1000
                aligned_RT[3, 3] = 1

                if save_path is not None:
                    coord_pts_rotated = aligned_RT[:3, :3] @ coord_pts.transpose() + aligned_RT[:3, 3:]
                    coord_pts_rotated = coord_pts_rotated.transpose()
                    np.savetxt(save_path + '_{}_{}_depth_pts.txt'.format(i, class_id), pts)
                    np.savetxt(save_path + '_{}_{}_coord_pts.txt'.format(i, class_id), coord_pts)
                    np.savetxt(save_path + '_{}_{}_coord_pts_aligned.txt'.format(i, class_id), coord_pts_rotated)

                if verbose:
                    print('Mask ID: ', i)
                    print('Scale: ', scales / 1000)
                    print('Rotation: ', rotation.transpose())
                    print('Translation: ', translation / 1000)

                # elapsed = time.time() - start
                # print('elapsed: ', elapsed)
                # elapses.append(elapsed)


            except Exception as e:
                message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(i + 1,
                                                                                            img_metas[0]['img_path'].split('/')[-2:], str(e))
                print(message)
                error_messages += message + '\n'
                aligned_RT = np.identity(4, dtype=np.float32)

                # print('Estimation takes {:03f}s.'.format(time.time() - start))
            # from camera world to computer vision frame
            z_180_RT = np.zeros((4, 4), dtype=np.float32)
            z_180_RT[:3, :3] = np.diag([-1, -1, 1])
            z_180_RT[3, 3] = 1

            # RTs[i, :, :] = z_180_RT @ aligned_RT
            RT = z_180_RT @ aligned_RT

            bbox_scales.append(bbox_scale)
            RTs.append(RT)

        cls_scales[i].append(np.array(bbox_scales))
        cls_RTs[i].append(np.array(RTs))

    return cls_RTs, cls_scales, error_messages, elapses


def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]


def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T # Transpose is the one that works

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    ScaleFact = 1/varP * np.sum(D) # scale factor
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)

    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

    OutTransform = np.identity(4)
    OutTransform[:3, :3] = ScaleMatrix @ Rotation
    OutTransform[:3, 3] = Translation

    # # Check
    # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    # Residual = np.linalg.norm(Diff[:3, :], axis=0)
    return Scales, Rotation, Translation, OutTransform
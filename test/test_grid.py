import numpy as np
import matplotlib.pyplot as plt
import os
grid_path = '/disk1/data/amodal_grip_dataset/data_test/voxel_grid/fffc15a1df2b43e3a838f3b43c3ee99e.npz'
scene_path = '/disk1/data/amodal_grip_dataset/data_test/scenes/fffc15a1df2b43e3a838f3b43c3ee99e.npz'
scene_process_path = '/disk1/data/amodal_grip_dataset/data_test/scenes_processed/fffc15a1df2b43e3a838f3b43c3ee99e.npz'

scene = dict(np.load(scene_path))
rgb = scene['rgb_imgs'][0]
plt.imshow(rgb)
plt.show()


scene_process = dict(np.load(scene_process_path))





grid = dict(np.load(grid_path))['grid'][0]
save_dir = './grid'
os.makedirs(save_dir, exist_ok=True)

for i in range(40):
    grid_x = grid[i, :, :]
    grid_y = grid[:, i, :]
    grid_z = grid[:, :, i]
    # plt.imshow(grid_x, cmap='coolwarm')
    plt.imsave(save_dir+f'/x{i}.png', grid_x)
    plt.imsave(save_dir+f'/y{i}.png', grid_y)
    plt.imsave(save_dir+f'/z{i}.png', grid_z)

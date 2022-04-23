import pandas as pd


csv = pd.read_csv('/hddisk2/data/hanyang/amodel_dataset/data_test/grasps.csv')
scene_id_unique = csv['scene_id'].unique()
grasp_infos = {i:[] for i in scene_id_unique}
for i in range(csv.__len__()):
    grasp_info = csv.iloc[i].to_list()
    grasp_infos[grasp_info[0]].append(grasp_info[1:-1])

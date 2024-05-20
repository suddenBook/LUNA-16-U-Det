import os
import shutil
import pandas as pd
from radio import CTImagesMaskedBatch
from radio.batchflow import Dataset, FilesIndex, Pipeline, F

# 配置数据集路径和输出路径
DATA_PATH = os.path.normpath('//vmware-host/Shared Folders/OneDrive/Final Project/Dataset/LUNA-16/')
OUT_PATH = os.path.normpath('//vmware-host/Shared Folders/OneDrive/Final Project/Main_Code_Paper/0-Code_Data/Data/')

# 确保输出路径存在
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, 'imgs'), exist_ok=True) 
os.makedirs(os.path.join(OUT_PATH, 'masks'), exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, 'split_lists'), exist_ok=True)

# 创建 LUNA16 数据集的 FilesIndex
luna_index = FilesIndex(path=os.path.join(DATA_PATH, 'subset', '*', '*.mhd'), no_ext=True)

# 创建 Dataset
luna_dataset = Dataset(index=luna_index, batch_class=CTImagesMaskedBatch)

anns = pd.read_csv(os.path.join(DATA_PATH, 'annotations.csv'))
anns = anns[['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']]
anns = anns.rename(columns={'seriesuid': 'series_uid', 
                            'coordX': 'x', 'coordY': 'y', 'coordZ': 'z',
                            'diameter_mm': 'diameter'})

print(anns)

# 定义预处理 pipeline
preprocessing = (
    Pipeline()
    .load(fmt='raw')
    .unify_spacing(shape=(128, 256, 256), spacing=(2.5, 1.0, 1.0))
    .fetch_nodules_info(anns)
    .create_mask()
    .dump(dst=os.path.join(OUT_PATH, 'imgs'), components='images')
    .dump(dst=os.path.join(OUT_PATH, 'masks'), components='masks')
)

# 运行预处理 pipeline
(luna_dataset >> preprocessing).run(batch_size=1)

# 划分数据集
luna_dataset.split(0.8, shuffle=True) 

# 保存划分后的数据集索引到 csv 文件
for i, part in enumerate(['train', 'test']):
    ids = list(getattr(luna_dataset, part).indices)
    with open(os.path.join(OUT_PATH, 'split_lists', f'{part}.csv'), 'w') as f:
        f.write('ids\n')
        f.write('\n'.join(ids))

print(f"Pre-Process Complete at {OUT_PATH}")
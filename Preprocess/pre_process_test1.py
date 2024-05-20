import os
import pandas as pd
from radio import batchflow as bf
from radio.batchflow import FilesIndex, Dataset
from radio import CTImagesMaskedBatch as CTIMB
from radio.pipelines import split_dump

# 设置路径
DATA_PATH = os.path.normpath('//vmware-host/Shared Folders/OneDrive/Final Project/Dataset/LUNA-16/')
SUBSET_PATH = os.path.join(DATA_PATH, 'subset')
OUT_PATH = os.path.normpath('//vmware-host/Shared Folders/OneDrive/Final Project/Main_Code_Paper/0-Code_Data/Data/')
anns = pd.read_csv(os.path.join(DATA_PATH, 'annotations.csv'))

# 修正 seriesuid 以匹配文件路径，并检查文件是否存在
valid_files = []
skipped_count = 0

for uid in anns['seriesuid']:
    file_path = os.path.normpath(os.path.join(SUBSET_PATH, uid + '.mhd'))
    if os.path.exists(file_path):
        valid_files.append(file_path)
    else:
        skipped_count += 1

print(f"Skipped {skipped_count} files because they were not found on disk.")

# 创建文件索引，确保 seriesuid 匹配文件名
luna_index = FilesIndex(path=valid_files, no_ext=False, sort=True)
lunaset = Dataset(index=luna_index, batch_class=CTIMB)

# 配置预处理参数
SPACING = (1.0, 1.0, 1.0)
SHAPE = (400, 512, 512)
PADDING = 'reflect'
METHOD = 'pil-simd'

kwargs_default = dict(shape=SHAPE, spacing=SPACING, padding=PADDING, method=METHOD)

# 配置和运行数据预处理 pipeline
crop_pipeline = split_dump(cancer_path=os.path.join(OUT_PATH, 'imgs'),
                           non_cancer_path=os.path.join(OUT_PATH, 'imgs'),
                           nodules=anns, fmt='blosc', nodule_shape=(32, 64, 64),
                           batch_size=20, **kwargs_default)

# 运行 pipeline
(lunaset >> crop_pipeline).run()

# 分割数据集
train_set, val_set, test_set = lunaset.split([0.7, 0.15, 0.15])

# 保存分割信息
train_set.indices.dump(os.path.join(OUT_PATH, 'split_lists', 'train_set.txt'))
val_set.indices.dump(os.path.join(OUT_PATH, 'split_lists', 'val_set.txt'))
test_set.indices.dump(os.path.join(OUT_PATH, 'split_lists', 'test_set.txt'))
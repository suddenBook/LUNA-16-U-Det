import os
import numpy as np
import pandas as pd
from radio import CTImagesBatch as CTIB
from radio.batchflow import FilesIndex, Dataset, Pipeline, F
from pathlib import Path

DATA_PATH = '//vmware-host/Shared Folders/OneDrive/Final Project/Dataset/LUNA-16/'
SUBSET_PATH = os.path.join(DATA_PATH, 'subset')
OUT_PATH = '//vmware-host/Shared Folders/OneDrive/Final Project/Main_Code_Paper/0-Code_Data/Data/'

os.makedirs(os.path.join(OUT_PATH, 'imgs'), exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, 'masks'), exist_ok=True)

anns = pd.read_csv(os.path.join(DATA_PATH, 'annotations.csv'))
anns = anns[['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']]
anns = anns.rename(columns={'seriesuid': 'series_uid', 
                            'coordX': 'x', 'coordY': 'y', 'coordZ': 'z',
                            'diameter_mm': 'diameter'})

SPACING = (1.0, 1.0, 1.0)
SHAPE = (400, 512, 512)
PADDING = 'reflect'
METHOD = 'pil-simd'

# Step-by-step pipeline
pipeline = Pipeline().load(fmt='raw')

# Create FilesIndex and Dataset
subset_path = Path(SUBSET_PATH)
mhd_files = [str(subset_path / f) for f in os.listdir(subset_path) if f.endswith('.mhd')]

lunaix = FilesIndex(path=mhd_files)
lunaset = Dataset(index=lunaix, batch_class=CTIB)

# Split dataset
lunaset.split(0.7)
if len(lunaset.train) < 10 or len(lunaset.test) < 10:
    raise ValueError("Dataset is too small to be split into train/test.")

print(len(lunaset.train))
print(len(lunaset.test))

# Run the pipeline step-by-step
try:
    (lunaset.train >> pipeline).run()
    print("Load step successful")
    
    pipeline = pipeline.unify_spacing(spacing=SPACING, shape=SHAPE, method=METHOD, padding=PADDING)
    (lunaset.train >> pipeline).run()
    print("Unify spacing step successful")
    
    pipeline = pipeline.normalize_hu(min_hu=-1200, max_hu=600)
    (lunaset.train >> pipeline).run()
    print("Normalize HU step successful")
    
    pipeline = pipeline.fetch_nodules_info(nodules=anns)
    (lunaset.train >> pipeline).run()
    print("Fetch nodules info step successful")
    
    pipeline = pipeline.create_mask()
    (lunaset.train >> pipeline).run()
    print("Create mask step successful")
    
    pipeline = pipeline.dump(dst=os.path.join(OUT_PATH, 'imgs'), components='images', fmt='blosc')
    (lunaset.train >> pipeline).run()
    print("Dump images step successful")
    
    pipeline = pipeline.dump(dst=os.path.join(OUT_PATH, 'masks'), components='masks', fmt='blosc')
    (lunaset.train >> pipeline).run()
    print("Dump masks step successful")
    
except Exception as e:
    print(f"Error: {e}")
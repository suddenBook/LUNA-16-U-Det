#!/bin/bash

#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:10:00
#SBATCH --mem=32G
#SBATCH --job-name=mhd_to_jpg

source /etc/profile

pip install --user --upgrade matplotlib PySide2 SimpleITK

python3 mhd_to_png.py
#!/bin/bash

#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 01:15:00
#SBATCH --mem=32G
#SBATCH --job-name=download_LUNA_dataset

source /etc/profile

python3 download_dataset.py

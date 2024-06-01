#!/bin/bash

#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:10:00
#SBATCH --mem=32G
#SBATCH --job-name=compare_MD5

source /etc/profile

find "../Dataset/" -type f -exec md5sum {} \; | sort > md5_hashes_temp.txt
sort md5_hashes.txt -o md5_hashes.txt
diff md5_hashes_temp.txt md5_hashes.txt
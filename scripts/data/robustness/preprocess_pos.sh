#!/bin/bash
#
#SBATCH --job-name=pos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

python preprocess_pos.py \
  --attack="phonetic" \
  --cpu_count=40
#!/bin/bash
#
#SBATCH --job-name=snli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

python preprocess_snli.py \
  --cpu_count=40 \
  --attack="phonetic"
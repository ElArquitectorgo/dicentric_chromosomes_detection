#!/usr/bin/env bash

#SBATCH -J launch.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20gb
#SBATCH --time=2:00:00

#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

#SBATCH --error=benchmark.%J.err
#SBATCH --output=benchmark.%J.out

module load python
conda activate yolo_v100

export MPLBACKEND=agg
export PYTORCH_ALLOC_CONF=expandable_segments:True

time python benchmark.py 

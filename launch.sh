#!/usr/bin/env bash

#SBATCH -J launch.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20gb
#SBATCH --time=24:00:00

#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1

#SBATCH --error=train.%J.err
#SBATCH --output=train.%J.out

module load python
conda activate yolo_v100

export MPLBACKEND=agg
export PYTORCH_ALLOC_CONF=expandable_segments:True

gpu_id=-1
echo Training raw_clahe_std raw_lt_std raw_lbp_std lt_clahe_std lbp_mean_std

for dataset in raw_clahe_std raw_lt_std raw_lbp_std lt_clahe_std lbp_mean_std; do
	for config in cfg/*; do
		time python train_val.py -d folds/train/$dataset -c $config -m x -g $gpu_id -n fr_${dataset}_x_`basename -s .yaml $config`
	done
done

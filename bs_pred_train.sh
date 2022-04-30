#!/bin/bash
#SBATCH --account=gpu
#SBATCH --partition=gpu
#SBATCH --job-name=bs_prediction_train
#SBATCH -o bs_prediction_train-%j.o
#SBATCH -e bs_prediction_train-%j.e
#SBATCH --nodelist=node022
#SBATCH --mem=100gb
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

srun python /data/hpcdata/users/kenzi22/mfdgp/bs_prediction_train.py
#!/bin/bash -l
#SBATCH --partition=swarm_a100 
#SBATCH --mem=100G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL


module load cuda/12.4.0      

conda activate ocrenv

echo 'conda environment loaded'

echo 'Training started for' $1

python3 cuda_check.py 

python3 $1 $2 $3

echo 'Training completed'


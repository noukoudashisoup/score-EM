#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=0.5_1.0_21
#SBATCH --output=slurm_output/%A 
source activate score_EM
python run_svgd.py --seed 21  --dist N  --sigma 1.0  --init_noise 1.0  --pi 0.5  --noise_interval 1  --mean 10  --inner_steps 3000  --alg spos  --noise_type decreasing

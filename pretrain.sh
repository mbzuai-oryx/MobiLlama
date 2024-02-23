#!/bin/sh
#SBATCH --job-name=mobillama
#SBATCH --account<account_name>
#SBATCH --partition=<partition>
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:8
#SBATCH -t 3-00:00:00

srun python main_mobillama.py --n_nodes 20 --run_wandb
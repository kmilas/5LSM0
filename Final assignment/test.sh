#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --time=00:25:00

srun apptainer exec --nv --env-file .env container.sif python3 test.py

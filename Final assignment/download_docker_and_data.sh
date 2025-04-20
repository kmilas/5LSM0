#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:06:00

# Pull container from dockerhub
#apptainer pull container.sif docker://cclaess/5lsm0:v1
apptainer pull container.sif docker://tjmjaspers/nncv2025:v7
# Use the huggingface-cli package inside the container to download the data
#mkdir -p data
#apptainer exec container.sif \
#    huggingface-cli download TimJaspersTue/5LSM0 --local-dir ./data --repo-type dataset

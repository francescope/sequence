#!/bin/bash
#SBATCH --job-name=cuda_job
#SBATCH --output=cuda_logs/cuda_job_%j.out
#SBATCH --error=cuda_logs/cuda_job_%j.err
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --time=01:00:00
#SBATCH --partition=students

# Diagnostica
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"

# Esegui con mpirun
mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda \
1048576 0.1 0.3 0.35 16384 32 2 16384 32 2 524288 262144 M 609823

# Basic test
#mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda \
#300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823

# Simple tests for race conditions
#mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda \
#1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435

# Simple tests for race conditions
#mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda \
#10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435

# Check that the program works for sequences longest than INT_MAX
#mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda \
#4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224

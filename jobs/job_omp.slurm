#!/bin/bash
#SBATCH --job-name=align_seq
#SBATCH --output=logs/omp_job.out
#SBATCH --error=logs/omp_job.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00  # Modifica il tempo massimo di esecuzione
#SBATCH --partition=students  # Sostituiscilo con la partizione del tuo cluster, se necessario
#SBATCH --export=ALL


# Imposta il numero di thread OpenMP
export OMP_NUM_THREADS=4

# Esegui il programma con srun
srun ./align_omp 4194304 0.1 0.3 0.35 2048 8 2 2048 32 2 524288 262144 M 609823

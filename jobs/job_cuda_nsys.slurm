#!/bin/bash
#SBATCH --job-name=profiling_job
#SBATCH --output=cuda_logs/profiling_%j.out
#SBATCH --error=cuda_logs/profiling_%j.err
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=students

ls -l /home/pennacchietti_1934929/1934929/align_cuda
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"

#Produce il file di profiling nsys
#nsys profile -o /home/pennacchietti_1934929/1934929/cuda_logs/profile_${SLURM_JOB_ID}_%p --trace=cuda,mpi --force-overwrite true \
#mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda 1048576 0.1 0.3 0.35 16384 32 2 16384 32 2 524288 262144 M 609823


#Profiling con focus su bilanciamento del carico
nsys profile -o /home/pennacchietti_1934929/1934929/cuda_logs/profileB_${SLURM_JOB_ID}_%p --trace=cuda,nvtx,mpi --force-overwrite true \
mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda 1048576 0.1 0.3 0.35 16384 32 2 16384 32 2 524288 262144 M 609823


# Step 3: Estrazione delle metriche con nv-nsight-cu-cli
echo "Raccogliendo le metriche con nv-nsight-cu-cli..."
nv-nsight-cu-cli --metrics achieved_occupancy,warp_execution_efficiency,gld_efficiency,gst_efficiency -o /home/pennacchietti_1934929/1934929/cuda_logs/metriche_${SLURM_JOB_ID}_%p
#Un unico file di profiling per tutti i processi
#nsys profile -o /home/pennacchietti_1934929/1934929/cuda_logs/profile_${SLURM_JOB_ID} --trace=cuda,mpi --force-overwrite true \
#mpirun -np 2 /home/pennacchietti_1934929/1934929/align_cuda_clean 67108864 0.1 0.3 0.35 32768 8 2 32768 8 2 524288 262144 M 609823

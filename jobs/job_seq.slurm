#!/bin/bash
#SBATCH --job-name=align_seq_test   # Nome del job
#SBATCH --output=logs/align_seq_%j.out  # File di output (con job ID)
#SBATCH --error=logs/align_seq_%j.err   # File di errore (con job ID)
#SBATCH --time=01:00:00      # Tempo massimo di esecuzione (modifica se necessario)
#SBATCH --partition=students # Cambia se serve una coda specifica
#SBATCH --ntasks=1           # Un solo task (sequenziale)
#SBATCH --cpus-per-task=1    # Numero di CPU da usare (modifica in base alle risorse)

#Esegui il programma con i parametri richiesti


#./align_seq 67108864 0.1 0.3 0.35 32768 8 2 32768 8 2 524288 262144 M 609823

#./align_seq 33554432 0.1 0.3 0.35 16384 8 2 16384 8 2 524288 262144 M 609823

#./align_seq 1048576 0.1 0.3 0.35 8192 32 2 8192 32 2 524288 262144 M 609823


./align_seq 32768 0.1 0.3 0.35 512 8 2 512 8 2 524288 262144 M 609823
./align_seq 65536 0.1 0.3 0.35 1024 8 2 1024 8 2 524288 262144 M 609823
./align_seq 131072 0.1 0.3 0.35 2048 8 2 2048 8 2 524288 262144 M 609823
./align_seq 262144 0.1 0.3 0.35 4096 8 2 4096 8 2 524288 262144 M 609823
./align_seq 524288 0.1 0.3 0.35 8192 16 2 8192 16 2 524288 262144 M 609823
./align_seq 1048576 0.1 0.3 0.35 16384 32 2 16384 32 2 524288 262144 M 609823

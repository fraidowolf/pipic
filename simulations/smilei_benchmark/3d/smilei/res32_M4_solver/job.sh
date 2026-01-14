#!/bin/bash
#
#SBATCH -o ./job_%j.out
#SBATCH -e ./job_%j.err
#SBATCH -J res32_smilei_benchmark
#SBATCH -t 72:00:00
#SBATCH -n 4
#SBATCH -c 32
#
#module load buildenv-gcc/2022a-eb
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
module load buildenv-intel/2023a-eb
module load HDF5/1.12.2-hpc1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_SCHEDULE=dynamic
mpprun ./smilei benchmark.py

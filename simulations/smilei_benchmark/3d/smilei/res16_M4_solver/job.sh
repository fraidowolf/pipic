#!/bin/bash
#
#SBATCH -o ./job_%j.out
#SBATCH -e ./job_%j.err
#SBATCH -J res16_smilei_benchmark
#SBATCH -t 24:00:00
#SBATCH -N 8
#
module load buildenv-gcc/2022a-eb
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
module load buildenv-intel/2023a-eb
module load HDF5/1.12.2-hpc1

mpprun ./smilei benchmark.py


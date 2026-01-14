#!/bin/bash
#
#SBATCH -o ./job_%j.out
#SBATCH -e ./job_%j.err
#SBATCH -J res8_longer_sim_box_smilei
#SBATCH -t 23:59:00
#SBATCH -N 1
#
module load buildenv-gcc/2022a-eb
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
module load buildenv-intel/2023a-eb
module load HDF5/1.12.2-hpc1

./smilei benchmark.py


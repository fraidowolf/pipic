#!/bin/bash
#
#SBATCH -o ./job_%j.out
#SBATCH -e ./job_%j.err
#SBATCH -J 0,5e10_hres
#SBATCH -t 4:00:00
#SBATCH -N 1
#
module load Python/3.10.4-env-hpc1-gcc-2022a-eb
source ~/.venv/pipic/bin/activate

python ./benchmark.py


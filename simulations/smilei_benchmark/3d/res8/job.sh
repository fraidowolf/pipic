#!/bin/bash
#
#SBATCH -o ./job_%j.out
#SBATCH -e ./job_%j.err
#SBATCH -J res8_pipic_benchmark
#SBATCH -t 120:00:00
#SBATCH -N 1
#
source ~/.venv/pipic_tutorial/bin/activate

python ./benchmark.py


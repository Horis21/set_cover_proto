#!/bin/bash

#SBATCH --job-name="bobotree"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-eemcs-bsc-ti

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-pandas
module load gurobi/11.0.1
module load py-mpi4py

srun python vertex_cover.py > output.log
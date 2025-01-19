#!/bin/bash

#SBATCH --job-name="bobotree"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=education-eemcs-bsc-ti

module load 2023r1
module load openmpi
module load py-scikit-learn
module load python
module load py-numpy
module load py-pandas
module load gurobi/11.0.1
module load py-mpi4py



# Ensure pip is installed and upgrade it
python -m ensurepip --upgrade

# Install required Python packages
python -m pip install --user gurobipy > pip_install.log 2>&1

# Check if pip install was successful
if [ $? -ne 0 ]; then
  echo "pip install failed. Check pip_install.log for details."
  exit 1
fi

# Run the Python script
srun python set-cover.py > output.log 2>&1

# Check if python script was successful
if [ $? -ne 0 ]; then
  echo "Python script failed. Check output.log for details."
  exit 2
fi

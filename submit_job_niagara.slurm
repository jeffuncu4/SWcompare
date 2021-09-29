#!/bin/bash
#SBATCH --account=rrg-ngrisoua
#SBATCH --mail-user=jeffrey.uncu@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=SWcompare
#SBATCH --output=output_SW_Lr0.25Ro0.1.txt
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=0-72:00 

module --force purge
module load CCEnv
module load StdEnv/2020 fftw-mpi mpi4py hdf5-mpi python

source $HOME/python_env/bin/activate

#source $HOME/dedalus/bin/activate

cd $SCRATCH/SWcompare/codes/

python run_experiments.py
deactivate

exit

#!/bin/bash
#SBATCH -J thrifty-optimize
#SBATCH -o logs/log-optimize.o%j
#SBATCH -e logs/log-optimize.e%j
#SBATCH -N 10
#SBATCH -t 32:00:00
#SBATCH -p cca
# --constraint=skylake

source ~/.bash_profile
init_conda
# init_env

cd /mnt/ceph/users/apricewhelan/projects/chemical-torus-imaging/scripts

date

mpirun python3 bootstrap_optimize.py --mpi 

date


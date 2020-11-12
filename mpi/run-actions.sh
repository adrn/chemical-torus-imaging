#!/bin/bash
#SBATCH -J thrifty
#SBATCH -o logs/log-actions.o%j
#SBATCH -e logs/log-actions.e%j
#SBATCH -N 2
#SBATCH -t 16:00:00
#SBATCH -p cca
# --constraint=skylake

source ~/.bash_profile
init_conda
# init_env

cd /mnt/ceph/users/apricewhelan/projects/chemical-torus-imaging/scripts

date

mpirun python3 compute_actions.py --mpi

date


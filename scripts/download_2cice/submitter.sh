#!/bin/bash
#SBATCH --job-name=cloudsat # Specify job name
#SBATCH --output=cloudsat.o%j # name for standard output log file
#SBATCH --error=cloudsat.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/HcModel/scripts/download_2cice/download.py $1 $2 
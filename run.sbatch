#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu_v100
#SBATCH --output=joblog/R-%x.%j.out
#SBATCH --error=joblog/R-%x.%j.err

# Load anything you want
module load cudnn/7.6.4-CUDA10.1

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=`netstat -tan | awk '$1 == "tcp" && $4 ~ /:/ { port=$4; sub(/^[^:]+:/, "", port); used[int(port)] = 1; } END { for (p = 10000; p <= 65535; ++p) if (! (p in used)) { print p; exit(0); }; exit(1); }'`

srun run.slurm.sh
#!/bin/bash
#SBATCH --partition=singlepe
#SBATCH --job-name=run_scale_factors
#SBATCH --output=run_scale_factors.log
#SBATCH --time=02:00:00
#SBATCH --account=mod3dev
#SBATCH --ntasks=1

module load intelpython

echo "start: $start_date"
echo "end: $end_date"
echo "iteration: $iteration"


python pcd.py
#python pcd_test.py

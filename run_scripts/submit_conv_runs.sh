#!/bin/bash

start_date=$1
end_date=$2
iteration=$3

run_dir=/work/MOD3DEV/jkumm/EMBER/CMAQ/36US3/run/

#scale_file=${run_dir}/run.cmaq54.36US3.2023firesv2.SCALE${iteration}_CONV_CHECK
assim_file=${run_dir}/run.cmaq54.36US3.2023firesv2.ASSIM${iteration}_CONV_CHECK

sbatch --export=start_date=$start_date,end_date=$end_date,iteration=$iteration $assim_file
#sbatch --export=start_date=$start_date,end_date=$end_date,iteration=$iteration $scale_file


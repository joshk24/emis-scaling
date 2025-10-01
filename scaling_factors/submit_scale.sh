#!/bin/bash

start_date=$1
end_date=$2
iteration=$3

if [[ -z "$start_date" || -z "$end_date" || -z "$iteration" ]]; then
	echo "Usage: $0 <start_date> <end_date> <iteration>"
	exit 1
fi

sbatch --export=start_date=$start_date,end_date=$end_date,iteration=$iteration run_scale.sh

#!/bin/bash

# Define probabilities, time limit, and total run count per instance
probabilities=(0.2 0.5 0.8)
time_limit=3600  # Adjust as needed
run_count=3

# Initialize command files
for i in {1..4}; do
  > commands$i.txt  # Clear or create the command files
done

# Populate command files with arguments for each instance
command_file_index=1
for instance_file in Tai*/Tai*_problem.txt; do
  for probability in "${probabilities[@]}"; do
    for run in $(seq 1 $run_count); do
      echo "python ../JSC_final.py /HOME/martigna/thesis/instance-files/$instance_file $probability $time_limit $run" >> commands$command_file_index.txt
      
      # Rotate to next command file (1 to 4) for balanced distribution
      ((command_file_index++))
      if [ $command_file_index -gt 4 ]; then
        command_file_index=1
      fi
    done
  done
done

echo "Command files generated as commands1.txt to commands4.txt."

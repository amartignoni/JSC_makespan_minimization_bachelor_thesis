#!/bin/bash

# Run commands in each file with four parallel jobs
for cmd_file in ../instance-files/commands1.txt; do
  (
    # Run each command in the file
    while IFS= read -r command; do
      $command &
      
      # Wait if 4 jobs are running, allowing only 4 at a time
      if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n  # Wait for one of the jobs to finish
      fi
    done < "$cmd_file"
    
    wait  # Ensure all commands in the current file complete
  ) &
done

wait  # Wait for all background jobs to complete
echo "All commands executed."

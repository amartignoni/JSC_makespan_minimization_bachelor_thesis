#!/bin/bash
#SBATCH --job-name=JLC-GA-Full-Run     # Job name
#SBATCH --mail-type=END,FAIL           # Mail events
#SBATCH --mail-user=<redacted>
#SBATCH --time=96:00:00                # Adjust depending on expected runtime
#SBATCH --output=JLC-GA-Full-Run_%j.log
#SBATCH --nodelist=<redacted>          # Only CPU node
#SBATCH --mincpus=128                  # Ask for enough threads

RUN_NUMBER=({1..5})
OUTPUT_DIR="genetic_results"
POP_SIZE=200
CROSSOVER="LOX"
MUTATION="SWAP"
INIT_METHOD="SEMI"
PROBS=(0.2 0.5 0.8)
ITERATIONS=(1 2 3)

MAX_PARALLEL_JOBS=128
job_pids=()

launch_job() {
  local run=$1
  local XX=$2
  local prob=$3
  local itr=$4

  instance="Tai${XX}_problem_${prob}_${itr}"

  uv run genetic_algorithm_runner.py \
    --problem_path "instance-files/Tai${XX}/Tai${XX}_problem.txt" \
    --cg_path "results/${instance}/cg_matrix.csv" \
    --lower_bounds_path "evaluation/${instance}/lower_bounds.csv" \
    --pop_size $POP_SIZE \
    --crossover $CROSSOVER \
    --mutation $MUTATION \
    --pmut 0.2 \
    --max_steps 1000000 \
    --max_no_improve 10000 \
    --output_dir $OUTPUT_DIR \
    --generate_gantt \
    --init_method $INIT_METHOD \
    --run_id $run &

  job_pids+=($!)
}

cleanup() {
  echo "Interrupt received, killing background jobs..."
  for pid in "${job_pids[@]}"; do
    kill $pid 2>/dev/null
  done
  wait "${job_pids[@]}" 2>/dev/null
  exit 1
}

trap cleanup SIGINT SIGTERM SIGKILL

for run in "${RUN_NUMBER[@]}"; do
  for i in {1..80}; do
    XX=$(printf "%02d" $i)
    for prob in "${PROBS[@]}"; do
      for itr in "${ITERATIONS[@]}"; do
        if [ ${#job_pids[@]} -lt $MAX_PARALLEL_JOBS ]; then
          launch_job $run $XX $prob $itr
        else
          wait -n "${job_pids[@]}"
          job_pids=($(jobs -p))
          launch_job $run $XX $prob $itr
        fi
      done
    done
  done
done

wait "${job_pids[@]}"

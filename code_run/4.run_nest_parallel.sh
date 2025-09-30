#!/bin/bash
# Define the molecule to test
MOLECULE="H2"
# Define the backend to test
BACKEND="ibm_brisbane"
# Date for the runs
DATE="03_27"
# Other parameters
ITERATIONS=72
CYCLES=6
RUNS_PER_EXECUTION=1
TOTAL_RUNS=10
# Create a log directory
LOG_DIR="ablation_logs_${DATE}"
mkdir -p "$LOG_DIR"

# Function to run experiments with specified schedules
run_experiment() {
    local schedules=("$@")
    local schedule_count=${#schedules[@]}
    local schedule_tag="schedules_${schedule_count}"
    
    # Calculate how many times we need to execute
    NUM_EXECUTIONS=$((TOTAL_RUNS / RUNS_PER_EXECUTION))
    echo "Will run with $schedule_count schedules, $RUNS_PER_EXECUTION run per execution, repeated $NUM_EXECUTIONS times"
    
    # Main loop for each execution
    for execution in $(seq 1 $NUM_EXECUTIONS); do
        # Define the results directory name base
        RESULTS_DIR_BASE="results_nest_parallel_${MOLECULE}_${DATE}_${BACKEND}_${schedule_tag}"
        COPY_DIR="${RESULTS_DIR_BASE}_execution${execution}"
        
        # Check if this specific execution has already been completed
        if [ -d "$COPY_DIR" ]; then
            echo "Skipping already completed execution $execution for $MOLECULE on $BACKEND with $schedule_count schedules"
            continue
        fi
        
        # Create log file
        safe_molecule=$(echo "$MOLECULE" | sed 's/+/P/g')
        log_file="${LOG_DIR}/${safe_molecule}_${BACKEND}_${schedule_tag}_execution${execution}.log"
        
        echo "Running execution: $execution of $NUM_EXECUTIONS" | tee -a "$log_file"
        echo "Molecule: $MOLECULE, Backend: $BACKEND, Schedules: ${schedules[*]}" | tee -a "$log_file"
        
        # Build the schedule parameters - use a single --schedules argument
        schedule_params="--schedules"
        for schedule in "${schedules[@]}"; do
            schedule_params+=" $schedule"
        done
        
        # Run the command and log output
        echo "Starting execution $execution at $(date)" | tee -a "$log_file"
        
        python 4.run_nest_parallel.py \
            $schedule_params \
            --iterations $ITERATIONS \
            --cycles $CYCLES \
            --num-runs $RUNS_PER_EXECUTION \
            --backend $BACKEND \
            --date=$DATE 2>&1 | tee -a "$log_file"
        
        # Find the results directory created by the script
        ORIGINAL_DIR=$(find . -maxdepth 1 -type d -name "results_nest_parallel_${MOLECULE}_${DATE}_${BACKEND}*" | sort | head -n 1)
        
        # Copy the results to a new directory with execution number if found
        if [ -n "$ORIGINAL_DIR" ] && [ -d "$ORIGINAL_DIR" ]; then
            echo "Copying results from $ORIGINAL_DIR to $COPY_DIR" | tee -a "$log_file"
            cp -r "$ORIGINAL_DIR" "$COPY_DIR"
            
            # Remove the original directory to free up space
            echo "Removing original results directory to free up space" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        else
            echo "WARNING: Expected results directory matching pattern results_nest_parallel_${MOLECULE}_${DATE}_${BACKEND}* not found!" | tee -a "$log_file"
        fi
        
        echo "Finished execution $execution at $(date)" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"
        
        # Sleep for a few seconds to allow resources to be released
        echo "Waiting for 5 seconds to release memory before next execution..."
        sleep 5
    done
}

# echo "Starting first set of experiments with 2 schedules (UP_FLAT UP_FLAT)"
# run_experiment "UP_FLAT" "UP_FLAT"

# echo "Waiting for 5 minutes before starting the next set of experiments..."
# sleep 300

# echo "Starting second set of experiments with 3 schedules (UP_FLAT UP_FLAT UP_FLAT)"
# run_experiment "UP_FLAT" "UP_FLAT" "UP_FLAT"

echo "Starting second set of experiments with 4 schedules (UP_FLAT UP_FLAT UP_FLAT UP_FLAT)"
run_experiment "UP_FLAT" "UP_FLAT" "UP_FLAT" "UP_FLAT"

echo "All runs completed!"
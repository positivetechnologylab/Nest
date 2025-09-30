#!/bin/bash
# Define the molecule to test - only H2
MOLECULES=("H2")
# Define the backend - only Brisbane
BACKEND="ibm_brisbane"
# Date for the runs
DATE="03_27"
# Other parameters
ITERATIONS=72
CYCLES=6
RUNS_PER_EXECUTION=1
TOTAL_RUNS=10

# Define schedules to run
SCHEDULES=("RELU")

# Function to determine qubit count for each molecule
get_qubit_count() {
    local molecule=$1
    case "$molecule" in
        "H2") echo 4 ;;
        "HeH+") echo 4 ;;
        "H3+") echo 6 ;;
        *) echo 4 ;; # Default fallback
    esac
}

# Function to set ESP range based on backend and qubit count
get_esp_range() {
    local backend=$1
    local qubits=$2
    
    # Default values if no specific match
    local min_esp=0.5
    local max_esp=0.7
    
    if [ "$qubits" -eq 4 ]; then
        case "$backend" in
            "ibm_strasbourg") min_esp=0.7; max_esp=0.95 ;;
            "ibm_sherbrooke") min_esp=0.7; max_esp=0.9 ;;
            "ibm_kyiv") min_esp=0.7; max_esp=0.9 ;;
            "ibm_brussels") min_esp=0.7; max_esp=0.95 ;;
            "ibm_brisbane") min_esp=0.7; max_esp=0.95 ;;
        esac
    elif [ "$qubits" -eq 6 ]; then
        case "$backend" in
            "ibm_strasbourg") min_esp=0.6; max_esp=0.85 ;;
            "ibm_sherbrooke") min_esp=0.6; max_esp=0.8 ;;
            "ibm_kyiv") min_esp=0.6; max_esp=0.75 ;;
            "ibm_brussels") min_esp=0.6; max_esp=0.85 ;;
            "ibm_brisbane") min_esp=0.6; max_esp=0.8 ;;
        esac
    fi
    
    echo "$min_esp $max_esp"
}

# Function to run one execution for a molecule with a specific schedule
run_schedule_execution() {
    local molecule=$1
    local schedule=$2
    local execution_num=$3
    
    # Create main directory for this schedule
    MAIN_DIR="complex_${molecule}_${schedule}"
    mkdir -p "$MAIN_DIR"
    
    # Create subdirectory for this specific run
    RUN_SUBDIR="${MAIN_DIR}/run_${execution_num}"
    mkdir -p "$RUN_SUBDIR"
    
    # Get qubit count for current molecule
    QUBIT_COUNT=$(get_qubit_count "$molecule")
    
    # Prepare log file name - replace + with P to avoid special character issues in filenames
    safe_molecule=$(echo "$molecule" | sed 's/+/P/g')
    
    # Determine ESP range based on backend and qubit count
    ESP_RANGE=($(get_esp_range "$BACKEND" "$QUBIT_COUNT"))
    MIN_ESP=${ESP_RANGE[0]}
    MAX_ESP=${ESP_RANGE[1]}
    
    # Create log directory if it doesn't exist
    LOG_DIR="ablation_logs_${DATE}"
    mkdir -p "$LOG_DIR"
    
    # Add a data directory environment variable to isolate molecule datasets
    export PENNYLANE_DATA_DIR="/tmp/pennylane_data_${molecule}_${schedule}_${execution_num}"
    mkdir -p "$PENNYLANE_DATA_DIR"
    
    # Run with specific schedule
    ORIGINAL_DIR="results_nest_single_${molecule}_${DATE}_${BACKEND}_${schedule}"
    
    # Check if this specific execution has already been completed
    if [ -d "$RUN_SUBDIR" ] && [ -n "$(ls -A "$RUN_SUBDIR")" ]; then
        echo "Skipping already completed $schedule execution for $molecule (run: $execution_num)"
    else
        log_file="${LOG_DIR}/${safe_molecule}_${BACKEND}_${schedule}_run${execution_num}.log"
        
        echo "Running with molecule: $molecule, backend: $BACKEND, schedule: $schedule, run: $execution_num" | tee -a "$log_file"
        echo "Molecule: $molecule, Qubits: $QUBIT_COUNT, Backend: $BACKEND" | tee -a "$log_file"
        echo "Using ESP range: [$MIN_ESP, $MAX_ESP]" | tee -a "$log_file"
        
        # Remove any existing original results directory to start fresh
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Removing existing results directory: $ORIGINAL_DIR" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        fi
        
        # Run the command and log output
        echo "Starting execution with $schedule at $(date)" | tee -a "$log_file"
        
        python 3.run_nest_single.py \
            --date "$DATE" \
            --backend "$BACKEND" \
            --schedule "$schedule" \
            --molecule "$molecule" \
            --iterations "$ITERATIONS" \
            --cycles "$CYCLES" \
            --min-esp "$MIN_ESP" \
            --max-esp "$MAX_ESP" \
            --num-runs "$RUNS_PER_EXECUTION" 2>&1 | tee -a "$log_file"
        
        # Check if the output directory exists and move to run subdirectory
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Moving results from $ORIGINAL_DIR to $RUN_SUBDIR" | tee -a "$log_file"
            cp -r "$ORIGINAL_DIR"/* "$RUN_SUBDIR"/
            
            # Save backend and schedule info
            echo "Backend: $BACKEND" > "${RUN_SUBDIR}/backend_info.txt"
            echo "Schedule: $schedule" >> "${RUN_SUBDIR}/backend_info.txt"
            echo "ESP Range: $MIN_ESP to $MAX_ESP" >> "${RUN_SUBDIR}/backend_info.txt"
            echo "Run number: $execution_num" >> "${RUN_SUBDIR}/backend_info.txt"
            
            # Remove the original directory to free up space
            echo "Removing original results directory to free up space" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        else
            echo "WARNING: Expected results directory $ORIGINAL_DIR not found!" | tee -a "$log_file"
        fi
        
        echo "Finished execution with $schedule at $(date)" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"
    fi
    
    echo "Completed run $execution_num for molecule $molecule with schedule $schedule"
}

# Calculate how many times we need to execute
NUM_EXECUTIONS=$TOTAL_RUNS
echo "Will run $NUM_EXECUTIONS executions for each schedule to achieve $TOTAL_RUNS total runs"

# Create log directory
LOG_DIR="ablation_logs_${DATE}"
mkdir -p "$LOG_DIR"

# Main loop for running all schedules with all executions - sequential
for molecule in "${MOLECULES[@]}"; do
    echo "Processing molecule: $molecule"
    
    for schedule in "${SCHEDULES[@]}"; do
        echo "Processing schedule: $schedule"
        
        # Create main directory for this schedule
        MAIN_DIR="complex_${molecule}_${schedule}"
        mkdir -p "$MAIN_DIR"
        
        # Save general info for this schedule run
        echo "Molecule: $molecule" > "${MAIN_DIR}/info.txt"
        echo "Schedule: $schedule" >> "${MAIN_DIR}/info.txt"
        echo "Backend: $BACKEND" >> "${MAIN_DIR}/info.txt"
        echo "Total runs: $TOTAL_RUNS" >> "${MAIN_DIR}/info.txt"
        echo "Date: $DATE" >> "${MAIN_DIR}/info.txt"
        
        for execution_num in $(seq 1 $NUM_EXECUTIONS); do
            echo "Starting run $execution_num of $NUM_EXECUTIONS for $molecule with schedule $schedule"
            run_schedule_execution "$molecule" "$schedule" "$execution_num"
            echo "Completed run $execution_num of $NUM_EXECUTIONS for $molecule with schedule $schedule"
        done
        
        echo "All runs completed for molecule $molecule with schedule $schedule"
    done
    
    echo "All schedules completed for molecule $molecule"
done

echo "All runs completed!"
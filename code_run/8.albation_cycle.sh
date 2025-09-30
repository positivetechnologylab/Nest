#!/bin/bash
# Define the molecule to test
MOLECULE="H2"
# Define the backends to choose from
BACKENDS=("ibm_brisbane" "ibm_kyiv" "ibm_brussels" "ibm_sherbrooke" "ibm_strasbourg")
# Randomly select a backend from the array
RANDOM_INDEX=$((RANDOM % ${#BACKENDS[@]}))
BACKEND="${BACKENDS[$RANDOM_INDEX]}"
# Date for the runs
DATE="03_27"
# Other parameters
SCHEDULE="UP_FLAT"
ITERATIONS=
# Cycles to test in the ablation study
# CYCLE_VALUES=(2 4 6 8 10)
CYCLE_VALUES=(4)
# MIN_ESP and MAX_ESP will be set dynamically based on molecule and backend
RUNS_PER_EXECUTION=1
TOTAL_RUNS=10

# Function to determine qubit count for the molecule
get_qubit_count() {
    local molecule=$1
    case "$molecule" in
        "H2") echo 4 ;;
        "HeH+") echo 4 ;;
        "H3+") echo 6 ;;
        "H4") echo 8 ;;
        "He2") echo 8 ;;
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
    elif [ "$qubits" -eq 8 ]; then
        case "$backend" in
            "ibm_strasbourg") min_esp=0.5; max_esp=0.6 ;;
            "ibm_sherbrooke") min_esp=0.5; max_esp=0.65 ;;
            "ibm_kyiv") min_esp=0.5; max_esp=0.6 ;;
            "ibm_brussels") min_esp=0.5; max_esp=0.7 ;;
            "ibm_brisbane") min_esp=0.5; max_esp=0.65 ;;
        esac
    fi
    
    echo "$min_esp $max_esp"
}

# Function to find the next execution number
find_next_execution() {
    local base_dir="$1"
    local max_exec=0
    
    # Look for existing execution directories and find the highest number
    for dir in "${base_dir}_execution"*; do
        if [ -d "$dir" ]; then
            # Extract the execution number from the directory name
            exec_num=$(echo "$dir" | grep -o "execution[0-9]*" | grep -o "[0-9]*")
            if [ -n "$exec_num" ] && [ "$exec_num" -gt "$max_exec" ]; then
                max_exec=$exec_num
            fi
        fi
    done
    
    # Return the next execution number
    echo $((max_exec + 1))
}

# Create a log directory
LOG_DIR="ablation_cycle_logs_${DATE}"
mkdir -p "$LOG_DIR"

# Calculate how many times we need to execute
NUM_EXECUTIONS=$((TOTAL_RUNS / RUNS_PER_EXECUTION))
echo "Will run with $RUNS_PER_EXECUTION run per execution, repeated $NUM_EXECUTIONS times to achieve $TOTAL_RUNS total runs for each cycle value"

# Get qubit count for molecule
QUBIT_COUNT=$(get_qubit_count "$MOLECULE")

# Display the randomly selected backend
echo "Randomly selected backend: $BACKEND"

# Determine ESP range based on backend and qubit count
ESP_RANGE=($(get_esp_range "$BACKEND" "$QUBIT_COUNT"))
MIN_ESP=${ESP_RANGE[0]}
MAX_ESP=${ESP_RANGE[1]}

echo "Molecule: $MOLECULE, Qubits: $QUBIT_COUNT, Backend: $BACKEND"
echo "Using ESP range: [$MIN_ESP, $MAX_ESP]"

# Main loop for ablation study - iterate through different cycle values
for CYCLES in "${CYCLE_VALUES[@]}"; do
    echo "========================================================"
    echo "Running ablation study for CYCLES = $CYCLES"
    echo "========================================================"
    
    # Calculate iterations based on cycles: 432 / CYCLES
    ITERATIONS=$((432 / CYCLES))
    echo "Using ITERATIONS = $ITERATIONS for CYCLES = $CYCLES"
    
    # Define the name of the original results directory that will be created
    # The python script doesn't include cycles in the directory name
    ORIGINAL_DIR="results_nest_single_${MOLECULE}_${DATE}_${BACKEND}_${SCHEDULE}"
    
    # Define the name of the directory where we'll store results with cycles info
    CYCLE_DIR="${ORIGINAL_DIR}_cycles${CYCLES}"
    
    # Find next execution number to resume from
    next_execution=$(find_next_execution "$CYCLE_DIR")
    echo "For $MOLECULE on $BACKEND with CYCLES=$CYCLES, found existing executions, will resume from execution $next_execution"
    
    # Check if we have already completed all executions
    if [ "$next_execution" -gt "$NUM_EXECUTIONS" ]; then
        echo "Skipping already completed job: $MOLECULE on $BACKEND with CYCLES=$CYCLES (all $NUM_EXECUTIONS executions exist)"
        continue
    fi
    
    # Prepare log file name - replace + with P to avoid special character issues in filenames
    safe_molecule=$(echo "$MOLECULE" | sed 's/+/P/g')
    
    echo "Starting from execution $next_execution through $NUM_EXECUTIONS"
    
    for execution in $(seq $next_execution $NUM_EXECUTIONS); do
        # Check if this specific execution has already been completed
        COPY_DIR="${CYCLE_DIR}_execution${execution}"
        if [ -d "$COPY_DIR" ]; then
            echo "Skipping already completed execution $execution for $MOLECULE on $BACKEND with CYCLES=$CYCLES"
            continue
        fi
        
        log_file="${LOG_DIR}/${safe_molecule}_${BACKEND}_cycles${CYCLES}_execution${execution}.log"
        
        echo "Running with molecule: $MOLECULE, backend: $BACKEND, cycles: $CYCLES, execution: $execution of $NUM_EXECUTIONS" | tee -a "$log_file"
        echo "Molecule: $MOLECULE, Qubits: $QUBIT_COUNT, Backend: $BACKEND" | tee -a "$log_file"
        echo "Using ESP range: [$MIN_ESP, $MAX_ESP]" | tee -a "$log_file"
        
        # Remove any existing original results directory to start fresh
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Removing existing results directory: $ORIGINAL_DIR" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        fi
        
        # Run the command and log output
        echo "Starting execution $execution with CYCLES=$CYCLES at $(date)" | tee -a "$log_file"
        
        python 3.run_nest_single.py \
            --date "$DATE" \
            --backend "$BACKEND" \
            --schedule "$SCHEDULE" \
            --molecule "$MOLECULE" \
            --iterations "$ITERATIONS" \
            --cycles "$CYCLES" \
            --min-esp "$MIN_ESP" \
            --max-esp "$MAX_ESP" \
            --num-runs "$RUNS_PER_EXECUTION" 2>&1 | tee -a "$log_file"
        
        # Copy the results to a new directory with cycle and execution info
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Copying results from $ORIGINAL_DIR to $COPY_DIR" | tee -a "$log_file"
            cp -r "$ORIGINAL_DIR" "$COPY_DIR"
            
            # Remove the original directory to free up space
            echo "Removing original results directory to free up space" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        else
            echo "WARNING: Expected results directory $ORIGINAL_DIR not found!" | tee -a "$log_file"
        fi
        
        echo "Finished execution $execution with CYCLES=$CYCLES at $(date)" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"
        
        # Sleep for a few seconds to allow resources to be released
        echo "Waiting for 5 seconds to release memory before next execution..."
        sleep 5
    done
done

echo "All cycle ablation runs completed!"
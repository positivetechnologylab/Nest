#!/bin/bash
# Define the molecules to test
# MOLECULES=("H2" "H3+" "HeH+")
# MOLECULES=("H2" "HeH+")
MOLECULES=("H3+")
# Define the backends to choose from randomly
ALL_BACKENDS=("ibm_brisbane" "ibm_kyiv" "ibm_brussels" "ibm_sherbrooke" "ibm_strasbourg")
# Date for the runs
DATE="03_27"
# Other parameters from your original command
ITERATIONS=72
CYCLES=6
# MIN_ESP and MAX_ESP will be set dynamically based on molecule and backend
RUNS_PER_EXECUTION=1
TOTAL_RUNS=30

# Define the starting index for complex folders
Start_Index=1

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

# Function to randomly select two different backends
# MODIFIED: Added bias against selecting Sherbrooke
select_random_backends() {
    local all_backends=("$@")
    local num_backends=${#all_backends[@]}
    
    # Special case to bias against Sherbrooke
    # First, attempt to find Sherbrooke in the array
    local sherbrooke_idx=-1
    for i in "${!all_backends[@]}"; do
        if [ "${all_backends[$i]}" = "ibm_sherbrooke" ]; then
            sherbrooke_idx=$i
            break
        fi
    done
    
    # 75% chance to exclude Sherbrooke from first selection if it exists
    if [ $sherbrooke_idx -ne -1 ] && [ $((RANDOM % 4)) -ne 0 ]; then
        # Remove Sherbrooke from consideration for first backend
        all_backends=("${all_backends[@]:0:$sherbrooke_idx}" "${all_backends[@]:$((sherbrooke_idx+1))}")
        local idx1=$((RANDOM % (num_backends-1)))
        local backend1=${all_backends[$idx1]}
        
        # Remove the first selected backend from the array for the second selection
        if [ $idx1 -lt $sherbrooke_idx ]; then
            sherbrooke_idx=$((sherbrooke_idx-1))  # Adjust index if we removed an element before it
        fi
        
        # Now decide if we want to include Sherbrooke in the possible second backends
        # 50% chance to include it back for second selection
        if [ $((RANDOM % 2)) -eq 0 ]; then
            # Add Sherbrooke back to the array at a random position
            local insert_pos=$((RANDOM % (num_backends-1)))
            all_backends=("${all_backends[@]:0:$insert_pos}" "ibm_sherbrooke" "${all_backends[@]:$insert_pos}")
        fi
        
        all_backends=("${all_backends[@]:0:$idx1}" "${all_backends[@]:$((idx1+1))}")
        local idx2=$((RANDOM % (${#all_backends[@]})))
        local backend2=${all_backends[$idx2]}
    else
        # Standard random selection
        local idx1=$((RANDOM % num_backends))
        local backend1=${all_backends[$idx1]}
        
        # Remove the first selected backend from the array for the second selection
        all_backends=("${all_backends[@]:0:$idx1}" "${all_backends[@]:$((idx1+1))}")
        local idx2=$((RANDOM % (num_backends-1)))
        local backend2=${all_backends[$idx2]}
    fi
    
    echo "$backend1 $backend2"
}

# The rest of the script remains unchanged
# Function to run one complete execution (UP_FLAT, optimal, QConcord) for a molecule
run_complete_execution() {
    local molecule=$1
    local execution_num=$2
    
    # Calculate the actual folder index using Start_Index
    local folder_index=$((Start_Index + execution_num - 1))
    echo "Starting complete execution $execution_num for molecule $molecule (folder index: $folder_index)"
    
    # Create unified results directory for this execution using the Start_Index-based numbering
    UNIFIED_DIR="complex_${folder_index}_${molecule}"
    mkdir -p "$UNIFIED_DIR"
    
    # Randomly select two backends
    SELECTED_BACKENDS=($(select_random_backends "${ALL_BACKENDS[@]}"))
    BACKEND=${SELECTED_BACKENDS[0]}
    BACKEND_2=${SELECTED_BACKENDS[1]}
    
    echo "Selected backends for execution $execution_num (folder index: $folder_index):"
    echo "  - $BACKEND (for UP_FLAT and optimal schedules)"
    echo "  - $BACKEND_2 (second backend for QConcord)"
    
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
    export PENNYLANE_DATA_DIR="/tmp/pennylane_data_${molecule}_${folder_index}"
    mkdir -p "$PENNYLANE_DATA_DIR"
    
    # 1. Run with UP_FLAT schedule
    SCHEDULE="UP_FLAT"
    ORIGINAL_DIR="results_nest_single_${molecule}_${DATE}_${BACKEND}_${SCHEDULE}"
    RESULTS_SUBDIR="${UNIFIED_DIR}/Nest"
    mkdir -p "$RESULTS_SUBDIR"
    
    # Check if this specific execution has already been completed
    if [ -d "$RESULTS_SUBDIR" ] && [ -n "$(ls -A "$RESULTS_SUBDIR")" ]; then
        echo "Skipping already completed UP_FLAT execution for $molecule (folder index: $folder_index)"
    else
        log_file="${LOG_DIR}/${safe_molecule}_${BACKEND}_${SCHEDULE}_execution${folder_index}.log"
        
        echo "Running with molecule: $molecule, backend: $BACKEND, schedule: $SCHEDULE, folder index: $folder_index" | tee -a "$log_file"
        echo "Molecule: $molecule, Qubits: $QUBIT_COUNT, Backend: $BACKEND" | tee -a "$log_file"
        echo "Using ESP range: [$MIN_ESP, $MAX_ESP]" | tee -a "$log_file"
        
        # Remove any existing original results directory to start fresh
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Removing existing results directory: $ORIGINAL_DIR" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        fi
        
        # Run the command and log output
        echo "Starting execution with $SCHEDULE at $(date)" | tee -a "$log_file"
        
        python 3.run_nest_single.py \
            --date "$DATE" \
            --backend "$BACKEND" \
            --schedule "$SCHEDULE" \
            --molecule "$molecule" \
            --iterations "$ITERATIONS" \
            --cycles "$CYCLES" \
            --min-esp "$MIN_ESP" \
            --max-esp "$MAX_ESP" \
            --num-runs "$RUNS_PER_EXECUTION" 2>&1 | tee -a "$log_file"
        
        # Check if the output directory exists and move to unified directory
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Moving results from $ORIGINAL_DIR to $RESULTS_SUBDIR" | tee -a "$log_file"
            cp -r "$ORIGINAL_DIR"/* "$RESULTS_SUBDIR"/
            
            # Save backend and schedule info
            echo "Backend: $BACKEND" > "${RESULTS_SUBDIR}/backend_info.txt"
            echo "Schedule: $SCHEDULE" >> "${RESULTS_SUBDIR}/backend_info.txt"
            echo "ESP Range: $MIN_ESP to $MAX_ESP" >> "${RESULTS_SUBDIR}/backend_info.txt"
            
            # Remove the original directory to free up space
            echo "Removing original results directory to free up space" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        else
            echo "WARNING: Expected results directory $ORIGINAL_DIR not found!" | tee -a "$log_file"
        fi
        
        echo "Finished execution with $SCHEDULE at $(date)" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"
    fi
    
    # 2. Run with optimal schedule
    SCHEDULE="optimal"
    ORIGINAL_DIR="results_nest_single_${molecule}_${DATE}_${BACKEND}_${SCHEDULE}"
    RESULTS_SUBDIR="${UNIFIED_DIR}/optimal"
    mkdir -p "$RESULTS_SUBDIR"
    
    # Check if this specific execution has already been completed
    if [ -d "$RESULTS_SUBDIR" ] && [ -n "$(ls -A "$RESULTS_SUBDIR")" ]; then
        echo "Skipping already completed optimal execution for $molecule (folder index: $folder_index)"
    else
        log_file="${LOG_DIR}/${safe_molecule}_${BACKEND}_${SCHEDULE}_execution${folder_index}.log"
        
        echo "Running with molecule: $molecule, backend: $BACKEND, schedule: $SCHEDULE, folder index: $folder_index" | tee -a "$log_file"
        echo "Molecule: $molecule, Qubits: $QUBIT_COUNT, Backend: $BACKEND" | tee -a "$log_file"
        echo "Using ESP range: [$MIN_ESP, $MAX_ESP]" | tee -a "$log_file"
        
        # Remove any existing original results directory to start fresh
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Removing existing results directory: $ORIGINAL_DIR" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        fi
        
        # Run the command and log output
        echo "Starting execution with $SCHEDULE at $(date)" | tee -a "$log_file"
        
        python 3.run_nest_single.py \
            --date "$DATE" \
            --backend "$BACKEND" \
            --schedule "$SCHEDULE" \
            --molecule "$molecule" \
            --iterations "$ITERATIONS" \
            --cycles "$CYCLES" \
            --min-esp "$MIN_ESP" \
            --max-esp "$MAX_ESP" \
            --num-runs "$RUNS_PER_EXECUTION" 2>&1 | tee -a "$log_file"
        
        # Check if the output directory exists and move to unified directory
        if [ -d "$ORIGINAL_DIR" ]; then
            echo "Moving results from $ORIGINAL_DIR to $RESULTS_SUBDIR" | tee -a "$log_file"
            cp -r "$ORIGINAL_DIR"/* "$RESULTS_SUBDIR"/
            
            # Save backend and schedule info
            echo "Backend: $BACKEND" > "${RESULTS_SUBDIR}/backend_info.txt"
            echo "Schedule: $SCHEDULE" >> "${RESULTS_SUBDIR}/backend_info.txt"
            echo "ESP Range: $MIN_ESP to $MAX_ESP" >> "${RESULTS_SUBDIR}/backend_info.txt"
            
            # Remove the original directory to free up space
            echo "Removing original results directory to free up space" | tee -a "$log_file"
            rm -rf "$ORIGINAL_DIR"
        else
            echo "WARNING: Expected results directory $ORIGINAL_DIR not found!" | tee -a "$log_file"
        fi
        
        echo "Finished execution with $SCHEDULE at $(date)" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"
    fi
    
    # 3. Run QConcord with both backends
    RESULTS_SUBDIR="${UNIFIED_DIR}/qconcord"
    mkdir -p "$RESULTS_SUBDIR"
    
    if [ -d "$RESULTS_SUBDIR" ] && [ -n "$(ls -A "$RESULTS_SUBDIR")" ]; then
        echo "Skipping already completed QConcord execution for $molecule (folder index: $folder_index)"
    else
        log_file="${LOG_DIR}/${safe_molecule}_qconcord_execution${folder_index}.log"
        echo "Starting QConcord for $molecule folder index $folder_index using backends $BACKEND and $BACKEND_2..." | tee -a "$log_file"
        
        # Save backend info
        echo "Backend1: $BACKEND" > "${RESULTS_SUBDIR}/backend_info.txt"
        echo "Backend2: $BACKEND_2" >> "${RESULTS_SUBDIR}/backend_info.txt"
        
        # Run QConcord with output directory option
        python 5.run_qoncord.py \
            --date $DATE \
            --molecule "$molecule" \
            --num_runs $RUNS_PER_EXECUTION \
            --backends $BACKEND $BACKEND_2 \
            --output_dir "$RESULTS_SUBDIR" 2>&1 | tee -a "$log_file"
            
        echo "Completed QConcord for $molecule folder index $folder_index" | tee -a "$log_file"
    fi
    
    echo "Completed full execution for molecule $molecule (folder index: $folder_index)"
    echo "All results saved in directory: $UNIFIED_DIR"
}

# Calculate how many times we need to execute
NUM_EXECUTIONS=$((TOTAL_RUNS / RUNS_PER_EXECUTION))
echo "Will run with $RUNS_PER_EXECUTION run per execution, repeated $NUM_EXECUTIONS times to achieve $TOTAL_RUNS total runs"
echo "Using Start_Index = $Start_Index for folder naming (complex_${Start_Index}_molecule, complex_$((Start_Index+1))_molecule, etc.)"

# Create log directory
LOG_DIR="ablation_logs_${DATE}"
mkdir -p "$LOG_DIR"

# Main loop for running all molecules with all executions - sequential
for molecule in "${MOLECULES[@]}"; do
    echo "Processing molecule: $molecule"
    
    for execution_num in $(seq 1 $NUM_EXECUTIONS); do
        folder_index=$((Start_Index + execution_num - 1))
        echo "Starting execution $execution_num of $NUM_EXECUTIONS for $molecule (folder index: $folder_index)"
        run_complete_execution "$molecule" "$execution_num"
        echo "Completed execution $execution_num of $NUM_EXECUTIONS for $molecule (folder index: $folder_index)"
    done
    
    echo "All executions completed for molecule $molecule"
done

echo "All runs completed!"
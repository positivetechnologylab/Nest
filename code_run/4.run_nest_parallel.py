from utils_parallel import *
import copy
from qiskit.circuit.library import EfficientSU2
from scipy.linalg import eigh
import argparse

MOLECULES = [
    ("H2", 0.742, "STO-3G"),
    ("H3+", 0.874, "STO-3G"),
    ("H4", 0.88, "STO-3G"),
    ("He2", 5.2, "6-31G"),
    ("HeH+", 0.775, "STO-3G")
]

def find_closest_esp_backend(backend_data: List[Dict], target_esp: float) -> Tuple[Dict, float]:
    """Find backend with ESP closest to target value."""
    min_diff = float('inf')
    closest_backend = None
    closest_esp = None
    
    for data in backend_data:
        diff = abs(data['esp'] - target_esp)
        if diff < min_diff:
            min_diff = diff
            closest_backend = data
            closest_esp = data['esp']
            
    return closest_backend, closest_esp

# Keep track of last used mapping for spider search
last_mapping = None

def spider_find(backend_data: List[Dict], full_data: Dict,
                target_esp: float, current_mapping: List[int], 
              coupling_map: List[List[int]], esp_threshold: float = 0.05,
              noise_dir: str = None, backend_name: str = None) -> Tuple[Dict, float]:
    new_mappings_added = 0
    def build_adjacency_map(qubit_map, coup_map):
        """Build adjacency map for the current mapping"""
        adj_map = {q: set() for q in qubit_map}
        for q1, q2 in coup_map:
            if q1 in qubit_map and q2 in qubit_map:
                adj_map[q1].add(q2)
                adj_map[q2].add(q1)
        return adj_map
    
    def find_leaf_nodes(adj_map):
        """Find leaf nodes (nodes with only one connection)"""
        return [node for node, neighbors in adj_map.items() if len(neighbors) == 1]
    
    def get_connected_candidates(coup_map, current_qubits):
        """Get valid candidates that are adjacent to at least two qubits in the current mapping"""
        # Build adjacency map for all qubits
        adj_map = {}
        for q1, q2 in coup_map:
            if q1 not in adj_map: adj_map[q1] = set()
            if q2 not in adj_map: adj_map[q2] = set()
            adj_map[q1].add(q2)
            adj_map[q2].add(q1)
        
        candidates = set()
        # Check all qubits not in current mapping
        for qubit in adj_map:
            if qubit not in current_qubits:
                # Count how many connections this qubit has to current mapping
                connections = sum(1 for neighbor in adj_map[qubit] if neighbor in current_qubits)
                if connections >= 1:  # Must be connected to at least 2 qubits in current mapping
                    candidates.add(qubit)
        return candidates
    
    mapping_set = frozenset(current_mapping)
    adj_map = build_adjacency_map(mapping_set, coupling_map)
    leaf_nodes = find_leaf_nodes(adj_map)
    print(f"Found leaf nodes: {leaf_nodes}")
    
    candidate_mappings_set = set()
    for leaf_node in leaf_nodes:
        # Find qubits that have multiple connections to current mapping
        connected_qubits = get_connected_candidates(coupling_map, mapping_set - {leaf_node})
        
        # Create new mappings by replacing leaf node with well-connected qubits
        temp_mapping = current_mapping.copy()
        leaf_idx = temp_mapping.index(leaf_node)
        
        for new_qubit in connected_qubits:
            new_mapping = temp_mapping.copy()
            new_mapping[leaf_idx] = new_qubit
            new_mapping_fs = frozenset(new_mapping)
            # Only add if it's not the current mapping
            if new_mapping_fs != mapping_set:
                candidate_mappings_set.add(new_mapping_fs)
    
    candidate_mappings = [list(m) for m in candidate_mappings_set]
    print(f"Generated {len(candidate_mappings)} unique candidate mappings")
    
    # Find matching backends for candidate mappings
    valid_backends = []
    full_backend_data = full_data['mappings_data']
    
    for mapping in candidate_mappings:
        matching_backend = None
        for backend in full_backend_data:
            if backend['mapping'] == mapping:
                matching_backend = backend
                break
            
        if matching_backend is None:
            circuit = EfficientSU2(num_qubits=len(mapping), reps=3)
            custom_backend = CustomBackend(
                noise_dir=noise_dir,
                qubits_mapping=mapping,
                circuit=circuit,
                backend_name=backend_name
            )
            
            esp = custom_backend.calculate_esp(circuit, mapping)
            
            new_backend = {
                'mapping': mapping,
                'backend': custom_backend,
                'transpiled_circuit': custom_backend.get_transpiled_circuit(),
                'esp': esp,
                'mapping_id': len(full_backend_data) + new_mappings_added + 1
            }
        
            full_backend_data.append(new_backend)
            new_mappings_added += 1
            
            # Track stats without writing to file
            original_length = len(full_data['mappings_data'])
            print(f"Length before adding: {original_length}")
            full_data['mappings_data'].append(new_backend)
            full_data['metadata']['num_mappings'] = len(full_data['mappings_data'])
            print(f"Length after adding: {len(full_data['mappings_data'])}")
            print(f"Added new mapping with ESP {esp:.4f} (no file write)")
            
            # Check ESP difference regardless of range
            esp_diff = abs(esp - target_esp)
            if esp_diff <= esp_threshold:
                valid_backends.append((new_backend, esp_diff))

        if matching_backend is not None:
            esp_diff = abs(matching_backend['esp'] - target_esp)
            if esp_diff <= esp_threshold:
                valid_backends.append((matching_backend, esp_diff))
                    
    print(f"New mappings added in this run: {new_mappings_added}")
    print(f"Total mappings after adding: {len(full_backend_data)}")
    
    if valid_backends:
        # Sort by ESP difference and return the closest match
        valid_backends.sort(key=lambda x: x[1])
        selected_backend = valid_backends[0][0]
        print(f"Found {len(valid_backends)} suitable mappings within ESP threshold {esp_threshold}. ")
        return selected_backend, selected_backend['esp']
    else:
        print(f"No suitable mappings found within ESP threshold {esp_threshold}. "
              f"Falling back to nearest ESP search.")
        # Fall back to original nearest ESP search
        min_diff = float('inf')
        closest_backend = None
        closest_esp = None
        
        for data in backend_data:
            diff = abs(data['esp'] - target_esp)
            if diff < min_diff:
                min_diff = diff
                closest_backend = data
                closest_esp = data['esp']
                    
        return closest_backend, closest_esp

def has_barrier_between_mappings(mapping1, mapping2, coupling_map):
    # Convert to sets for easier operations
    set1 = set(mapping1)
    set2 = set(mapping2)
    
    # Build an adjacency map from the coupling map
    adjacency = {}
    for q1, q2 in coupling_map:
        if q1 not in adjacency:
            adjacency[q1] = set()
        if q2 not in adjacency:
            adjacency[q2] = set()
        adjacency[q1].add(q2)
        adjacency[q2].add(q1)
    
    # Check if any qubit in mapping1 is directly connected to any qubit in mapping2
    for q1 in set1:
        if q1 in adjacency:
            for q2 in adjacency[q1]:
                if q2 in set2:
                    # Found a direct connection
                    return False
    
    return True

def find_mapping_with_barrier(backend_data, first_mapping, target_esp, coupling_map):
    # Sort backends by ESP difference from target
    sorted_backends = sorted(backend_data, key=lambda b: abs(b['esp'] - target_esp))
    
    # Find first mapping that has a barrier
    for backend in sorted_backends:
        mapping = backend['mapping']
        if has_barrier_between_mappings(first_mapping, mapping, coupling_map):
            return backend, backend['esp']
    
    # If no mapping with barrier found, return the closest ESP one and warn
    print("WARNING: No mapping with barrier found. Using closest ESP mapping instead.")
    return find_closest_esp_backend(backend_data, target_esp)


def run_parallel_mapping_experiment(
    ansatz: QuantumCircuit,
    full_data: List[Dict],
    schedule_types: List[str],  # List of schedule types, one for each parallel job
    iterations_per_mapping: int = 35,
    min_esp: float = 0.3,
    max_esp: float = 0.9,
    n_cycles: int = 12,
    coupling_map_static: List[List[int]] = None,
    noise_dir: str = None,
    backend_name: str = None,
    molecule: str = "H2"
) -> Dict:
    global last_mapping
    last_mapping = None  # Reset for this run
    backend_data = full_data['mappings_data']
    
    # Number of parallel jobs is determined by the length of schedule_types
    n_jobs = len(schedule_types)
    
    # Create ESP schedulers for each job
    schedulers = []
    esp_schedules = []
    
    for schedule_type in schedule_types:
        scheduler = ESPScheduler(min_esp, max_esp, n_cycles)
        esp_schedule = scheduler.get_schedule(n_cycles, (min_esp, max_esp), schedule_type)
        schedulers.append(scheduler)
        esp_schedules.append(esp_schedule)
    
    # Filter backends based on ESP range
    valid_backends = [b for b in backend_data if min_esp <= b['esp'] <= max_esp]
    if len(valid_backends) == 0:
        print(f"WARNING: No mappings found with ESP between {min_esp} and {max_esp}")
        valid_backends = backend_data
    
    # Initialize results
    results = {
        'schedule_types': schedule_types,
        'esp_schedules': esp_schedules
    }
    
    # Create result structures for each mapping
    for i in range(n_jobs):
        results[f'mapping{i+1}_results'] = []
        results[f'mapping{i+1}_history'] = {
            'target_esp_schedule': esp_schedules[i],
            'actual_esp_schedule': [],
            'selected_mappings': []
        }
    
    # Find molecule data from MOLECULES list
    molecule_data = None
    for mol in MOLECULES:
        if mol[0] == molecule:
            molecule_data = mol
            break
    
    if molecule_data is None:
        raise ValueError(f"Molecule '{molecule}' not found in MOLECULES list")
    
    # Load molecular Hamiltonian using the selected molecule
    print(f"Loading molecule: {molecule_data[0]}, bond length: {molecule_data[1]}, basis: {molecule_data[2]}")
    dataset = qml.data.load("qchem", molname=molecule_data[0], bondlength=molecule_data[1], basis=molecule_data[2])[0]
    hamiltonian, qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)
    print(f"qubits: {qubits}")
    hamiltonian_matrix = qml.matrix(hamiltonian)
    eigenvalues, eigenvectors = eigh(hamiltonian_matrix)
    ground_energy = eigenvalues[0]
    print(f"Ground state energy (from diagonalization): {ground_energy:.8f} Hartree")
    print("\nFirst 5 energy eigenvalues:")
    for i, energy in enumerate(eigenvalues[:5]):
        print(f"Energy level {i}: {energy:.8f} Hartree")
        
    hamiltonian_qiskit = to_qiskit_hamiltonian(hamiltonian)

    # Initialize VQE runners for each mapping (will update circuits during run)
    vqe_runners = [None] * n_jobs
    initial_params = [None] * n_jobs
    
    # We'll track the last mapping used for each schedule
    last_mappings = [None] * n_jobs
    
    for cycle in range(n_cycles):
        # Get target ESP values for this cycle
        target_esp_values = [esp_schedules[i][cycle] for i in range(n_jobs)]
        
        print(f"\nStarting cycle {cycle + 1}/{n_cycles}")
        for i, schedule_type in enumerate(schedule_types):
            print(f"Schedule {i+1} ({schedule_type}) target ESP: {target_esp_values[i]:.4f}")
        
        # Lists to store the selected backends and mappings for this cycle
        selected_backends = []
        selected_mappings = []
        selected_esp_values = []
        
        # Select first mapping
        if schedule_types[0] == 'base':
            idx = np.random.randint(0, len(valid_backends))
            backend1 = valid_backends[idx]
            esp1 = backend1['esp']
        else:
            if last_mappings[0] is not None:
                backend1, esp1 = spider_find(
                    backend_data=valid_backends,
                    full_data=full_data,
                    target_esp=target_esp_values[0],
                    current_mapping=last_mappings[0],
                    coupling_map=coupling_map_static,
                    noise_dir=noise_dir,
                    backend_name=backend_name
                )
            else:
                backend1, esp1 = find_closest_esp_backend(valid_backends, target_esp_values[0])
        mapping1 = backend1['mapping']
        last_mappings[0] = mapping1
        selected_backends.append(backend1)
        selected_mappings.append(mapping1)
        selected_esp_values.append(esp1)
        
        # For each remaining mapping (2 to n), ensure it has barriers with all previous mappings
        for i in range(1, n_jobs):
            suitable_backends = []
            for backend in valid_backends:
                mapping = backend['mapping']
                # Check if mapping has barriers with all previously selected mappings
                has_barriers_with_all = True
                for prev_mapping in selected_mappings:
                    if not has_barrier_between_mappings(prev_mapping, mapping, coupling_map_static):
                        has_barriers_with_all = False
                        break
                
                if has_barriers_with_all:
                    suitable_backends.append(backend)
            
            if not suitable_backends:
                print(f"WARNING: No mappings with barrier found for mapping {i+1}!")
                print(f"Using all backends for mapping {i+1} selection.")
                suitable_backends = valid_backends
            
            if schedule_types[i] == 'base':
                idx = np.random.randint(0, len(suitable_backends))
                selected_backend = suitable_backends[idx]
                selected_esp = selected_backend['esp']
            else:
                if last_mappings[i] is not None:
                    selected_backend, selected_esp = spider_find(
                        backend_data=suitable_backends,
                        full_data=full_data,
                        target_esp=target_esp_values[i],
                        current_mapping=last_mappings[i],
                        coupling_map=coupling_map_static,
                        noise_dir=noise_dir,
                        backend_name=backend_name
                    )
                else:
                    selected_backend, selected_esp = find_closest_esp_backend(suitable_backends, target_esp_values[i])
            
            selected_mapping = selected_backend['mapping']
            last_mappings[i] = selected_mapping
            selected_backends.append(selected_backend)
            selected_mappings.append(selected_mapping)
            selected_esp_values.append(selected_esp)
        
        # Print selected mappings
        print(f"Selected mappings:")
        for i in range(n_jobs):
            print(f"Mapping {i+1}: {selected_mappings[i]} with ESP={selected_esp_values[i]:.4f}")
        
        # Update history in results
        for i in range(n_jobs):
            results[f'mapping{i+1}_history']['actual_esp_schedule'].append(selected_esp_values[i])
            results[f'mapping{i+1}_history']['selected_mappings'].append(selected_mappings[i])
        
        # Initialize or update VQE runners
        for i in range(n_jobs):
            if vqe_runners[i] is None:
                vqe_runners[i] = VQERunner(
                    hamiltonian=hamiltonian_qiskit,
                    ansatz=ansatz,
                    custom_backend=selected_backends[i],
                    shots=1024
                )
            else:
                vqe_runners[i].transpiled_circuit = selected_backends[i]['transpiled_circuit']
                simulator = AerSimulator(
                    noise_model=selected_backends[i]['backend'].noise_model,
                    coupling_map=selected_backends[i]['backend'].coupling_map,
                    basis_gates=selected_backends[i]['backend'].basis_gates,
                    shots=1024
                )
                vqe_runners[i].estimator = BackendEstimatorV2(backend=simulator)
        
        # Run optimization for each mapping
        for i in range(n_jobs):
            result = vqe_runners[i].optimize(
                initial_params=initial_params[i],
                max_iter=iterations_per_mapping,
                optimizer_type='COBYLA',
                current_cycle=cycle,
                n_cycles=n_cycles
            )
            results[f'mapping{i+1}_results'].append(result)
            
            # Track the best parameters from this cycle
            if cycle > 0:
                cycle_energies = vqe_runners[i].optimization_history['energies'][-iterations_per_mapping:]
                cycle_params = vqe_runners[i].optimization_history['parameters'][-iterations_per_mapping:]
                
                # Find index of minimum energy in this cycle
                best_idx = cycle_energies.index(min(cycle_energies))
                # Use the parameters that gave the best energy in this cycle
                initial_params[i] = np.array(cycle_params[best_idx])
                print(f"Mapping {i+1} Results: final energy = {result.fun:.6f}")
                print(f"Mapping {i+1} Best energy in cycle = {min(cycle_energies):.6f}")
            else:
                # For the first cycle, just use the result parameters
                initial_params[i] = result.x
                print(f"Mapping {i+1} Results: final energy = {result.fun:.6f}")
    
    # Add VQE runners to results for plotting
    for i in range(n_jobs):
        results[f'vqe_runner{i+1}'] = vqe_runners[i]
    
    return results

def run_multi_experiment(
    ansatz: QuantumCircuit,
    full_data: Dict,
    schedule_types: List[str],  # List of schedule types, size determines parallel jobs
    n_runs: int = 5,
    iterations_per_mapping: int = 35,
    min_esp: float = 0.3,
    max_esp: float = 0.9,
    n_cycles: int = 12,
    coupling_map_static: List[List[int]] = None,
    output_dir: str = "multi_run_results",
    noise_dir: str = None,
    backend_name: str = None,
    molecule: str = "H2"
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of parallel jobs
    n_jobs = len(schedule_types)
    
    aggregated_results = {
        'schedule_types': schedule_types,
        'min_esp': min_esp,
        'max_esp': max_esp,
        'n_cycles': n_cycles,
        'n_runs': n_runs,
        'molecule': molecule,
        'individual_runs': [],
        'overall_stats': {
            'mean_run_time': 0.0,
            'total_run_time': 0.0
        }
    }
    
    # Create statistics structure for each mapping
    for i in range(n_jobs):
        aggregated_results[f'mapping{i+1}_stats'] = {
            'mean_energy_per_cycle': [],
            'min_energy_per_cycle': [],
            'mean_final_energy': 0.0,
            'min_final_energy': 0.0
        }
    
    for run_idx in range(n_runs):
        print(f"\n======= Starting Run {run_idx + 1}/{n_runs} =======")
        
        start_time = datetime.now()
        results = run_parallel_mapping_experiment(
            ansatz=ansatz,
            full_data=full_data,
            schedule_types=schedule_types,
            iterations_per_mapping=iterations_per_mapping,
            min_esp=min_esp,
            max_esp=max_esp,
            n_cycles=n_cycles,
            coupling_map_static=coupling_map_static,
            noise_dir=noise_dir,
            backend_name=backend_name,
            molecule=molecule
        )
        run_time = (datetime.now() - start_time).total_seconds()
        results['run_time'] = run_time
        results['run_idx'] = run_idx + 1
        
        # Save individual run results
        run_specific_dir = os.path.join(output_dir, f"run_{run_idx + 1}")
        os.makedirs(run_specific_dir, exist_ok=True)
        # Save results
        results_file = save_parallel_mapping_results(results, run_specific_dir)
        print(f"\nResults saved to: {results_file}")
        
        # Add results to aggregated data
        aggregated_results['individual_runs'].append(results)
    
    # Calculate statistics across all runs
    calculate_multi_run_statistics(aggregated_results)
    
    # Save aggregated results
    schedule_str = "_".join(schedule_types)
    aggregated_file = os.path.join(output_dir, f"aggregated_results_{schedule_str}_{molecule}.pkl")
    
    # Create a copy without the individual_runs to save space
    save_results = copy.deepcopy(aggregated_results)
    # Keep only essential information from individual runs to save space
    save_results['individual_runs'] = []
    
    for run in aggregated_results['individual_runs']:
        run_info = {
            'run_idx': run['run_idx'],
            'run_time': run['run_time']
        }
        
        # Add final energy for each mapping
        for i in range(n_jobs):
            if f'vqe_runner{i+1}' in run:
                run_info[f'mapping{i+1}_final_energy'] = run[f'vqe_runner{i+1}'].optimization_history['energies'][-1]
            else:
                run_info[f'mapping{i+1}_final_energy'] = None
        
        save_results['individual_runs'].append(run_info)
    
    with open(aggregated_file, 'wb') as f:
        pickle.dump(save_results, f)
    
    return aggregated_results

def calculate_multi_run_statistics(aggregated_results: Dict):
    """Calculate statistics across multiple runs, tracking energies per iteration"""
    n_runs = aggregated_results['n_runs']
    n_cycles = aggregated_results['n_cycles']
    iterations_per_mapping = 35  # From global variable in the code
    
    # Determine the number of parallel jobs
    n_jobs = len(aggregated_results['schedule_types'])
    
    # Initialize arrays to collect data across runs
    final_energies = [[] for _ in range(n_jobs)]
    run_times = []
    
    # Find the maximum number of iterations across all runs for each mapping
    max_iterations = [0] * n_jobs
    
    for run_data in aggregated_results['individual_runs']:
        for i in range(n_jobs):
            vqe_runner = run_data[f'vqe_runner{i+1}']
            max_iterations[i] = max(max_iterations[i], len(vqe_runner.optimization_history['energies']))
    
    # Initialize arrays for per-iteration energy statistics
    energies_by_iteration = []
    for i in range(n_jobs):
        energies_by_iteration.append([[] for _ in range(max_iterations[i])])
    
    # Collect data across all runs
    for run_data in aggregated_results['individual_runs']:
        # Collect run times
        run_times.append(run_data['run_time'])
        
        # Process each mapping
        for i in range(n_jobs):
            vqe_runner = run_data[f'vqe_runner{i+1}']
            
            # Collect final energies
            final_energies[i].append(vqe_runner.optimization_history['energies'][-1])
            
            # Collect energies by iteration
            energies = vqe_runner.optimization_history['energies']
            
            for iter_idx in range(len(energies)):
                if iter_idx < len(energies_by_iteration[i]):
                    energies_by_iteration[i][iter_idx].append(energies[iter_idx])
    
    # Calculate statistics for each mapping
    for i in range(n_jobs):
        mapping_stats = aggregated_results[f'mapping{i+1}_stats']
        
        # Calculate final energy statistics
        mapping_stats['mean_final_energy'] = np.mean(final_energies[i])
        mapping_stats['min_final_energy'] = np.min(final_energies[i])
        mapping_stats['max_final_energy'] = np.max(final_energies[i])
        
        # Initialize per-iteration statistics
        mapping_stats['mean_energy_per_iteration'] = []
        mapping_stats['min_energy_per_iteration'] = []
        mapping_stats['max_energy_per_iteration'] = []
        
        # Calculate statistics by iteration
        for iter_energies in energies_by_iteration[i]:
            if iter_energies:
                mapping_stats['mean_energy_per_iteration'].append(np.mean(iter_energies))
                mapping_stats['min_energy_per_iteration'].append(np.min(iter_energies))
                mapping_stats['max_energy_per_iteration'].append(np.max(iter_energies))
            else:
                mapping_stats['mean_energy_per_iteration'].append(None)
                mapping_stats['min_energy_per_iteration'].append(None)
                mapping_stats['max_energy_per_iteration'].append(None)
        
        # Keep cycle-based statistics for backward compatibility
        mapping_stats['mean_energy_per_cycle'] = []
        mapping_stats['min_energy_per_cycle'] = []
        
        # Store iteration counts for plotting
        mapping_stats['iterations_count'] = max_iterations[i]
    
    # Calculate overall statistics
    aggregated_results['overall_stats']['mean_run_time'] = np.mean(run_times)
    aggregated_results['overall_stats']['total_run_time'] = sum(run_times)

def save_parallel_mapping_results(results: Dict, output_dir: str) -> str:
    """Save results from parallel mapping experiment to a pickle file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    schedule_str = "_".join(results['schedule_types'])
    filename = f"parallel_mapping_{schedule_str}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    return filepath

def main(schedule_types: List[str],
         mappings_file: str = None,
         n_runs: int = 5,
         schedule_min_esp: float = 0.3,
         schedule_max_esp: float = 0.9,
         output_dir: str = "multi_run_results",
         noise_dir: str = None,
         backend_name: str = None,
         iterations_per_mapping: int = 36,
         n_cycles: int = 12,
         date: str = "03_27",
         num_qubits: int = 4,
         molecule: str = "H2"):
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('qiskit').setLevel(logging.WARNING)
    
    # Determine mappings path and noise directory
    if mappings_file:
        mappings_path = mappings_file
    else:
        if not date or not backend_name:
            raise ValueError("Either mappings_file or both date and backend must be provided")
        mappings_path = f"maps/{date}/{num_qubits}/{date}/{backend_name}/mappings.pkl"
        
    print(f"Mappings path: {mappings_path}")
    
    # Determine noise directory
    if not noise_dir:
        if not date or not backend_name:
            noise_dir = "noise_data"
        else:
            noise_dir = f"noise_models/{date}/{backend_name}"
    print(f"Noise directory: {noise_dir}")
    
    # Load backend data
    full_data, backend_data, meta_data = load_backend_data(mappings_path)
    print(f"\nLoaded mappings from {mappings_path}:")
    print(f"Number of mappings: {len(full_data['mappings_data'])}, ESP range: \
          [{meta_data['esp_statistics']['min_esp']}, {meta_data['esp_statistics']['max_esp']}]")
    
    # Load noise data
    noise_model, coupling_map_full, basis_gates, _, _ = load_noise_data(noise_dir)
    
    # Create ansatz circuit
    ansatz = EfficientSU2(num_qubits=num_qubits, reps=3)
    
    print(f"\nStarting parallel mapping VQE experiment with:")
    print(f"- Schedules: {schedule_types}")
    print(f"- Number of parallel mappings: {len(schedule_types)}")
    print(f"- Molecule: {molecule}")
    print(f"- ESP Range: [{schedule_min_esp}, {schedule_max_esp}]")
    print(f"- Iterations per mapping: {iterations_per_mapping}")
    print(f"- Cycles: {n_cycles}")
    
    # Run the multi-mapping experiment
    start_time = datetime.now()
    results = run_multi_experiment(
        ansatz=ansatz,
        full_data=full_data,
        schedule_types=schedule_types,
        n_runs=n_runs,
        iterations_per_mapping=iterations_per_mapping,
        min_esp=schedule_min_esp,
        max_esp=schedule_max_esp,
        n_cycles=n_cycles,
        coupling_map_static=coupling_map_full,
        output_dir=output_dir,
        noise_dir=noise_dir,
        backend_name=backend_name,
        molecule=molecule
    )
    run_time = (datetime.now() - start_time).total_seconds()
    results['run_time'] = run_time

if __name__ == "__main__":
    VALID_SCHEDULES = [
        'LR', 'CR', 'LT', 'LTT', 'CT', 'CTT', 
        'RR', 'ER', 'RTV', 'RTH', 'ETV', 'ETH',
        'base', 'optimal', 'UP_FLAT'
    ]
    
    VALID_MOLECULES = [mol[0] for mol in MOLECULES]
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel schedule VQE experiment')
    parser.add_argument('--schedules', type=str, nargs='+', required=True,
                        help='Schedule types to use. The number of schedules determines the number of parallel jobs')
    parser.add_argument('--mappings-file', type=str, default=None,
                        help='Path to the mappings file (if not provided, will be constructed from date and backend)')
    parser.add_argument('--date', type=str, required=False, help='Date in format MM_DD (e.g., 03_26)')
    parser.add_argument('--backend', type=str, required=False, help='Backend name (e.g., ibm_brisbane)')
    parser.add_argument('--iterations', type=int, default=36, 
                        help='Number of iterations per mapping')
    parser.add_argument('--cycles', type=int, default=12, 
                        help='Number of cycles to run')
    parser.add_argument('--min-esp', type=float, default=0.704, 
                        help='Minimum ESP value')
    parser.add_argument('--max-esp', type=float, default=1.0, 
                        help='Maximum ESP value')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of times to run the experiment')
    parser.add_argument('--output-dir', type=str, default='multi_run_results',
                        help='Directory to save results')
    parser.add_argument('--noise-dir', type=str, default='noise_data',
                        help='Directory containing noise data')
    parser.add_argument('--num-qubits', type=int, default=4,
                        help='Number of qubits in the ansatz')
    parser.add_argument('--molecule', type=str, default='H2', choices=VALID_MOLECULES,
                        help='Molecule to use for the experiment')
    
    args = parser.parse_args()
    
    # Validate that all provided schedules are valid
    for schedule in args.schedules:
        if schedule not in VALID_SCHEDULES:
            raise ValueError(f"Invalid schedule type: {schedule}. Valid options are: {VALID_SCHEDULES}")
    
    main(
        schedule_types=args.schedules,
        mappings_file=args.mappings_file,
        n_runs=args.num_runs,
        schedule_min_esp=args.min_esp,
        schedule_max_esp=args.max_esp,
        output_dir=args.output_dir,
        noise_dir=args.noise_dir,
        backend_name=args.backend if args.backend else None,
        iterations_per_mapping=args.iterations,
        n_cycles=args.cycles,
        date=args.date,
        num_qubits=args.num_qubits,
        molecule=args.molecule
    )
    
    # Example commands:
    # Running with 3 parallel mappings:
    # python run_parallel.py --schedules UP_FLAT LR CR --date 03_27 --backend ibm_strasbourg --iterations 72 --cycles 6 --num-runs 1 --molecule H2
    # 
    # Running with 5 parallel mappings:
    # python run_parallel.py --schedules UP_FLAT LR CR CT RR --mappings-file mappings_exhausted.pkl --noise-dir noise_data --iterations 36 --cycles 12 --num-runs 5 --output-dir test_results --molecule HeH+
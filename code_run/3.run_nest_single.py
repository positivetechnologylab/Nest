from utils import *
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import pickle
from typing import List, Dict, Tuple
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit import QuantumCircuit
from scipy.linalg import eigh

MOLECULES = [
    ("H2", 0.742, "STO-3G"),
    ("H3+", 0.874, "STO-3G"),
    ("H4", 0.88, "STO-3G"),
    ("He2", 5.2, "6-31G"),
    ("HeH+", 0.775, "STO-3G")
]

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

def run_single_mapping_experiment(
    ansatz: QuantumCircuit,
    full_data: Dict,
    schedule_type: str,
    molecule_name: str = "H2",
    iterations_per_mapping: int = 35,
    min_esp: float = 0.3,
    max_esp: float = 0.9,
    n_cycles: int = 12,
    coupling_map_static: List[List[int]] = None,
    hamiltonian: SparsePauliOp = None,
    noise_dir: str = None,
    backend_name: str = None
) -> Dict:
    backend_data = full_data['mappings_data']
    # Filter backends based on ESP range
    valid_backends = [b for b in backend_data if min_esp <= b['esp'] <= max_esp]
    if len(valid_backends) == 0:
        print(f"WARNING: No mappings found with ESP between {min_esp} and {max_esp}")
        valid_backends = backend_data

    hamiltonian_qiskit = to_qiskit_hamiltonian(hamiltonian)

    ################### Run optimal strategy ###################
    results = {
        'schedule_type': schedule_type,
        'mapping_results': [],
        'esp_schedule': [],
        'mapping_history': {
            'target_esp_schedule': [],
            'actual_esp_schedule': [],
            'selected_mappings': []
        }
    }
    if schedule_type == 'optimal':
        # Find the backend with the highest ESP
        best_backend = max(valid_backends, key=lambda x: x['esp'])
        best_esp = best_backend['esp']
        best_mapping = best_backend['mapping']
        print(f"\nUsing optimal strategy with best ESP mapping")
        print(f"Selected mapping: {best_mapping} with ESP={best_esp:.4f}")
        results['esp_schedule'] = [best_esp] * n_cycles
        results['mapping_history']['target_esp_schedule'] = [best_esp] * n_cycles
        results['mapping_history']['selected_mappings'] = [best_mapping] * n_cycles

        
        # Initialize VQE runner with the best backend
        vqe_runner = VQERunner(
            hamiltonian=hamiltonian_qiskit,
            ansatz=ansatz,
            custom_backend=best_backend['backend'],
            shots=1024
        )
        
        total_iterations = iterations_per_mapping * n_cycles
        print(f"Running single optimization for {total_iterations} iterations")
        
        result = vqe_runner.optimize(
            initial_params=None,
            max_iter=total_iterations,
            optimizer_type='COBYLA',
            rhobeg=1.0
        )
        
        results['mapping_results'].append(result)
        results['vqe_runner'] = vqe_runner
        return results

    ################### Run Nest strategy ###################
    
    scheduler = ESPScheduler(min_esp, max_esp, n_cycles)
    esp_schedule = scheduler.get_schedule(n_cycles, (min_esp, max_esp), schedule_type)
    
    # Initialize results
    results = {
        'schedule_type': schedule_type,
        'mapping_results': [],
        'esp_schedule': esp_schedule,
        'mapping_history': {
            'target_esp_schedule': esp_schedule,
            'actual_esp_schedule': [],
            'selected_mappings': []
        }
    }

    # Initialize VQE runner (will update circuits during run)
    vqe_runner = None
    initial_params = None
    
    # Track the last mapping used
    last_mapping = None

    # For random mapping
    idx_rm = np.random.randint(0, len(valid_backends))
    backend_info_rm = valid_backends[idx_rm]
    esp_rm = backend_info_rm['esp']
    
    for cycle in range(n_cycles):
        target_esp = esp_schedule[cycle]
        print(f"\nStarting cycle {cycle + 1}/{n_cycles}")
        print(f"Schedule ({schedule_type}) target ESP: {target_esp:.4f}")
        
        # Select mapping based on schedule type
        if last_mapping is not None:
            backend_info, esp = spider_find(
                backend_data=valid_backends,
                full_data=full_data,
                target_esp=target_esp,
                current_mapping=last_mapping,
                coupling_map=coupling_map_static,
                noise_dir=noise_dir,
                backend_name=backend_name
            )
        else:
            # Find backend with closest ESP to target
            backend_info, esp = find_closest_esp_backend(valid_backends, target_esp)
        
        mapping = backend_info['mapping']
        last_mapping = mapping
        
        print(f"Selected mapping: {mapping} with ESP={esp:.4f}")
        results['mapping_history']['actual_esp_schedule'].append(esp)
        results['mapping_history']['selected_mappings'].append(mapping)
        
        # Initialize or update VQE runner
        if vqe_runner is None:
            vqe_runner = VQERunner(
                hamiltonian=hamiltonian_qiskit,
                ansatz=ansatz,
                custom_backend=backend_info['backend'],
                shots=1024
            )
        else:
            vqe_runner.transpiled_circuit = backend_info['transpiled_circuit']
            simulator = AerSimulator(
                noise_model=backend_info['backend'].noise_model,
                coupling_map=backend_info['backend'].coupling_map,
                basis_gates=backend_info['backend'].basis_gates,
                shots=1024
            )
            vqe_runner.estimator = BackendEstimatorV2(backend=simulator)
        
        
        result = vqe_runner.optimize(
            initial_params=initial_params,
            max_iter=iterations_per_mapping,
            optimizer_type='COBYLA',
            current_cycle=cycle,
            n_cycles=n_cycles
        )
        results['mapping_results'].append(result)
        
        # Track the best parameters from this cycle
        if cycle > 0:
            cycle_energies = vqe_runner.optimization_history['energies'][-iterations_per_mapping:]
            cycle_params = vqe_runner.optimization_history['parameters'][-iterations_per_mapping:]
            
            # Find index of minimum energy in this cycle
            best_idx = cycle_energies.index(min(cycle_energies))
            # Use the parameters that gave the best energy in this cycle
            initial_params = np.array(cycle_params[best_idx])

            print(f"Results: final energy = {result.fun:.6f}")
            print(f"Best energy in cycle = {min(cycle_energies):.6f}")
        else:
            # For the first cycle, just use the result parameters
            initial_params = result.x
            print(f"Results: final energy = {result.fun:.6f}")
    
    # Add VQE runner to results for plotting
    results['vqe_runner'] = vqe_runner
    return results

def save_single_mapping_results(results: Dict, output_dir: str, date: str, backend_name: str) -> str:
    """Save the results from a single mapping experiment to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{backend_name}_{date}_{results['schedule_type']}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Create a copy to avoid modifying the original
    save_results = copy.deepcopy(results)
    
    # Get energy history and find best energy
    energies = results['vqe_runner'].optimization_history['energies']
    best_energy = min(energies)
    best_energy_idx = energies.index(best_energy)
    
    # Calculate which cycle and iteration within cycle achieved the best energy
    n_cycles = len(results['mapping_results'])
    iterations_per_mapping = len(energies) // n_cycles if n_cycles > 0 else len(energies)
    
    best_cycle = best_energy_idx // iterations_per_mapping + 1
    best_iteration_in_cycle = best_energy_idx % iterations_per_mapping + 1
    
    # Add metadata about the run
    save_results['metadata'] = {
        'date': date,
        'backend_name': backend_name,
        'timestamp': timestamp,
        'schedule_type': results['schedule_type'],
        'n_cycles': n_cycles,
        'iterations_per_mapping': iterations_per_mapping,
        'best_energy': best_energy,
        'best_energy_iteration': best_energy_idx + 1,  # 1-indexed for user readability
        'best_energy_cycle': best_cycle,
        'best_energy_iteration_in_cycle': best_iteration_in_cycle,
        'final_energy': energies[-1]
    }
    
    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(save_results, f)
    
    # Also save a JSON summary with the key results
    energies_with_iterations = [{"iteration": i+1, "energy": e} for i, e in enumerate(energies)]
    
    # Calculate energy improvements per cycle
    cycle_improvements = []
    for cycle in range(n_cycles):
        start_idx = cycle * iterations_per_mapping
        end_idx = min((cycle + 1) * iterations_per_mapping, len(energies))
        
        if start_idx < len(energies) and end_idx <= len(energies):
            cycle_start_energy = energies[start_idx]
            cycle_end_energy = energies[end_idx - 1]
            cycle_best_energy = min(energies[start_idx:end_idx])
            cycle_best_idx = energies.index(cycle_best_energy, start_idx, end_idx)
            cycle_best_iteration = cycle_best_idx + 1  # Global iteration number (1-indexed)
            if results['schedule_type'] != 'optimal':
                improvement = {
                    "cycle": cycle + 1,
                    "esp": results['mapping_history']['actual_esp_schedule'][cycle],
                    "mapping": results['mapping_history']['selected_mappings'][cycle],
                    "start_energy": cycle_start_energy,
                    "end_energy": cycle_end_energy,
                    "best_energy": cycle_best_energy,
                    "best_iteration": cycle_best_iteration,
                    "improvement": cycle_start_energy - cycle_best_energy
                }
            else:
                improvement = {
                    "cycle": cycle + 1,
                    "esp": results['esp_schedule'][cycle],
                    "mapping": results['mapping_history']['selected_mappings'][cycle],
                }
            cycle_improvements.append(improvement)
    
    summary = {
        'date': date,
        'backend_name': backend_name,
        'schedule_type': results['schedule_type'],
        'run_time': results.get('run_time', 0),
        'final_energy': energies[-1],
        'best_energy': best_energy,
        'best_energy_iteration': best_energy_idx + 1,
        'best_energy_cycle': best_cycle,
        'best_energy_iteration_in_cycle': best_iteration_in_cycle,
        'actual_esp_values': results['mapping_history']['actual_esp_schedule'],
        'target_esp_values': results['mapping_history']['target_esp_schedule'],
        'timestamp': timestamp,
        'n_cycles': n_cycles,
        'iterations_per_mapping': iterations_per_mapping,
        'energies_per_iteration': energies_with_iterations,
        'cycle_improvements': cycle_improvements
    }
    
    json_path = os.path.join(output_dir, f"{backend_name}_{date}_{results['schedule_type']}_{timestamp}_summary.json")
    with open(json_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    # Create a more readable text summary
    if results['schedule_type'] != 'optimal':
        txt_path = os.path.join(output_dir, f"{backend_name}_{date}_{results['schedule_type']}_{timestamp}_summary.txt")
        with open(txt_path, 'w') as f:
            f.write(f"VQE Experiment Summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Date: {date}\n")
            f.write(f"Backend: {backend_name}\n")
            f.write(f"Schedule type: {results['schedule_type']}\n")
            f.write(f"Run time: {results.get('run_time', 0):.2f} seconds\n\n")
            
            f.write(f"Results\n")
            f.write(f"-------\n")
            f.write(f"Final energy: {energies[-1]:.8f}\n")
            f.write(f"Best energy: {best_energy:.8f} (iteration {best_energy_idx + 1}, cycle {best_cycle}, iteration {best_iteration_in_cycle} in cycle)\n\n")
            
            f.write(f"Cycle Information\n")
            f.write(f"----------------\n")
            for i, improvement in enumerate(cycle_improvements):
                f.write(f"Cycle {i+1}: ESP={improvement['esp']:.4f}, Mapping={improvement['mapping']}\n")
                f.write(f"  Start energy: {improvement['start_energy']:.8f}\n")
                f.write(f"  End energy: {improvement['end_energy']:.8f}\n")
                f.write(f"  Best energy: {improvement['best_energy']:.8f} (global iteration {improvement['best_iteration']})\n")
                f.write(f"  Improvement: {improvement['improvement']:.8f}\n\n")
    
    return filepath

def plot_single_mapping_results(results: Dict, output_dir: str, date: str, backend_name: str):
    """Create plots for the results of a single mapping experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    schedule_type = results['schedule_type']
    
    # Get energy values from VQE runner
    vqe_runner = results['vqe_runner']
    energies = vqe_runner.optimization_history['energies']
    iterations = list(range(1, len(energies) + 1))
    
    # Dynamically determine iterations_per_mapping from the results
    n_cycles = len(results['mapping_results'])
    iterations_per_mapping = len(energies) // n_cycles if n_cycles > 0 else len(energies)
    
    # 1. Plot energies by iteration
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, 'b-', linewidth=2)
    
    # Mark cycle boundaries with vertical lines
    for cycle in range(1, n_cycles):
        plt.axvline(x=cycle * iterations_per_mapping, color='gray', linestyle='--', alpha=0.5)
        # Add cycle number labels
        plt.text(cycle * iterations_per_mapping + 5, min(energies) + 0.02 * (max(energies) - min(energies)), 
                 f'Cycle {cycle+1}', verticalalignment='bottom')
    
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title(f'Energy per Iteration: {backend_name} ({date}) - {schedule_type}')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'energy_per_iteration_{schedule_type}.png'), dpi=300)
    plt.close()
    
    # 2. Plot target vs actual ESP
    if results['schedule_type'] != 'optimal':
        plt.figure(figsize=(10, 6))
    
        cycles = list(range(1, len(results['mapping_history']['target_esp_schedule']) + 1))
        target_esp = results['mapping_history']['target_esp_schedule']
        actual_esp = results['mapping_history']['actual_esp_schedule']
        
        
        plt.plot(cycles, target_esp, 'b-', label='Target ESP')
        plt.plot(cycles, actual_esp, 'r--', label='Actual ESP')
        plt.scatter(cycles, actual_esp, color='red', s=50)
        
        plt.xlabel('Cycle')
        plt.ylabel('ESP')
        plt.title(f'Target vs Actual ESP: {backend_name} ({date}) - {schedule_type}')
        plt.xticks(cycles)  # Show all cycle numbers
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'esp_schedule_{schedule_type}.png'), dpi=300)
        plt.close()
    
        # 3. Plot selected mappings for each cycle
        plt.figure(figsize=(12, 6))
        
        selected_mappings = results['mapping_history']['selected_mappings']
        colors = plt.cm.viridis(np.linspace(0, 1, n_cycles))
        
        for i, mapping in enumerate(selected_mappings):
            plt.scatter([i+1] * len(mapping), mapping, c=[colors[i]], marker='o', s=80, label=f'Cycle {i+1}')
            
            # Connect points in the same mapping with lines
            for j in range(len(mapping)):
                plt.text(i+1.1, mapping[j], str(j), fontsize=9)
        
        plt.xlabel('Cycle')
        plt.ylabel('Qubit Index')
        plt.title(f'Selected Qubit Mappings: {backend_name} ({date}) - {schedule_type}')
        plt.yticks(range(0, max([max(m) for m in selected_mappings])+1))
        plt.xticks(range(1, n_cycles+1))
        plt.grid(True)
        
        # If there aren't too many cycles, add a legend
        if n_cycles <= 8:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'selected_mappings_{schedule_type}.png'), dpi=300)
        plt.close()
    
        # 4. Plot energy convergence within each cycle
        plt.figure(figsize=(12, 8))
        
        for cycle in range(n_cycles):
            start_idx = cycle * iterations_per_mapping
            end_idx = min((cycle + 1) * iterations_per_mapping, len(energies))
            cycle_energies = energies[start_idx:end_idx]
            cycle_iterations = list(range(1, len(cycle_energies) + 1))
            
            plt.subplot(int(np.ceil(n_cycles/2)), 2, cycle+1)
            plt.plot(cycle_iterations, cycle_energies, 'o-', linewidth=1.5, markersize=3, label=f'Cycle {cycle+1}')
            plt.xlabel('Iteration within cycle')
            plt.ylabel('Energy')
            plt.title(f'Cycle {cycle+1} (ESP: {results["mapping_history"]["actual_esp_schedule"][cycle]:.4f})')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'energy_per_cycle_{schedule_type}.png'), dpi=300)
        plt.close()

def main(schedule_type: str,
         date: str,
         backend_name: str,
         molecule: str = "H2",
         iterations_per_mapping: int = 35,
         n_cycles: int = 12,
         schedule_min_esp: float = 0.3,
         schedule_max_esp: float = 0.9,
         output_dir: str = None,
         num_qubits: int = 4,
         num_runs: int = 1):
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('qiskit').setLevel(logging.WARNING)
    
    # Find molecule in MOLECULES list
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
    hamiltonian, num_qubits = dataset.hamiltonian, len(dataset.hamiltonian.wires)
    print(f"qubits: {num_qubits}")
    hamiltonian_matrix = qml.matrix(hamiltonian)
    eigenvalues, eigenvectors = eigh(hamiltonian_matrix)
    ground_energy = eigenvalues[0]
    print(f"Ground state energy (from diagonalization): {ground_energy:.8f} Hartree")
    print("\nFirst 5 energy eigenvalues:")
    for i, energy in enumerate(eigenvalues[:5]):
        print(f"Energy level {i}: {energy:.8f} Hartree")

    mappings_path = f"maps/{date}/{num_qubits}/{date}/{backend_name}/mappings.pkl"
    print(f"Mappings path: {mappings_path}")
    noise_dir = f"noise_models/{date}/{backend_name}"
    print(f"Noise directory: {noise_dir}")
    base_output_dir = f"results_nest_single_{molecule}_{date}_{backend_name}_{schedule_type}"
    print(f"Base output directory: {base_output_dir}")
    
    # Check if paths exist
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings file not found at: {mappings_path}")
    
    if not os.path.exists(noise_dir):
        raise FileNotFoundError(f"Noise directory not found at: {noise_dir}")
    
    # Load backend data
    full_data, backend_data, meta_data = load_backend_data(mappings_path)
    print(f"\nLoaded mappings from {mappings_path}:")
    print(f"Number of mappings: {len(full_data['mappings_data'])}, ESP range: \
          [{meta_data['esp_statistics']['min_esp']}, {meta_data['esp_statistics']['max_esp']}]")

    # Load noise data
    noise_model, coupling_map_static, basis_gates, _, _ = load_noise_data(noise_dir)
    
    # Create ansatz circuit
    ansatz = EfficientSU2(num_qubits=num_qubits, reps=3)
    
    print(f"\nStarting {num_runs} runs of single-mapping VQE experiment with:")
    print(f"- Schedule: {schedule_type}")
    print(f"- Molecule: {molecule}")
    print(f"- ESP Range: [{schedule_min_esp}, {schedule_max_esp}]")
    print(f"- Iterations per mapping: {iterations_per_mapping}")
    print(f"- Cycles: {n_cycles}")
    
    # Run multiple experiments
    for run_index in range(num_runs):
        print(f"\nStarting run {run_index + 1}/{num_runs}")
        run_output_dir = os.path.join(base_output_dir, f"run_{run_index}")
        
        # Run the experiment
        start_time = datetime.now()
        results = run_single_mapping_experiment(
            ansatz=ansatz,
            full_data=full_data,
            schedule_type=schedule_type,
            molecule_name=molecule,
            iterations_per_mapping=iterations_per_mapping,
            min_esp=schedule_min_esp,
            max_esp=schedule_max_esp,
            n_cycles=n_cycles,
            coupling_map_static=coupling_map_static,
            hamiltonian=hamiltonian,
            noise_dir=noise_dir,
            backend_name=backend_name
        )
        run_time = (datetime.now() - start_time).total_seconds()
        results['run_time'] = run_time
        
        # Save results for this run
        results_file = save_single_mapping_results(results, run_output_dir, date, backend_name)
        print(f"\nResults saved to: {results_file}")
        
        # Plot results for this run
        # if results['schedule_type'] != 'optimal':
        plot_single_mapping_results(results, run_output_dir, date, backend_name)
    
    # Create aggregate plots and statistics
    if num_runs > 1:
        print("\nCreating aggregate plots and statistics...")
        plot_multi_run_comparison(base_output_dir, num_runs, date, backend_name, schedule_type)
        compile_multi_run_stats(base_output_dir, num_runs, date, backend_name, schedule_type)
        print(f"Aggregate results saved to: {os.path.join(base_output_dir, 'aggregate')}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    VALID_SCHEDULES = [
        'LR', 'CR', 'LT', 'LTT', 'CT', 'CTT', 
        'RR', 'ER', 'RTV', 'RTH', 'ETV', 'ETH',
        'base', 'optimal', 'UP_FLAT','LR_single', 'CR_single', 'LT_single', 'LTT_single', 'CT_single', 'CTT_single', 
        'RR_single', 'ER_single', 'RTV_single', 'RTH_single', 'ETV_single', 'ETH_single', 'RELU'
    ]
    
    
    VALID_MOLECULES = [mol[0] for mol in MOLECULES]
    
    # # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run single schedule VQE experiment')
    parser.add_argument('--date', type=str, required=True, help='Date in format MM_DD (e.g., 03_26)')
    parser.add_argument('--backend', type=str, required=True, help='Backend name (e.g., ibm_brisbane)')
    parser.add_argument('--schedule', type=str, default='UP_FLAT', choices=VALID_SCHEDULES, 
                        help='Schedule type to use')
    parser.add_argument('--molecule', type=str, default='H2', choices=VALID_MOLECULES,
                        help='Molecule to simulate')
    parser.add_argument('--iterations', type=int, default=35, 
                        help='Number of iterations per mapping')
    parser.add_argument('--cycles', type=int, default=12, 
                        help='Number of cycles to run')
    parser.add_argument('--min-esp', type=float, default=0.704, 
                        help='Minimum ESP value')
    parser.add_argument('--max-esp', type=float, default=1.0, 
                        help='Maximum ESP value')
    parser.add_argument('--num-qubits', type=int, default=4, 
                        help='Number of qubits')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of times to run the experiment')
    
    args = parser.parse_args()
    
    main(
        schedule_type=args.schedule,
        date=args.date,
        backend_name=args.backend,
        molecule=args.molecule,
        iterations_per_mapping=args.iterations,
        n_cycles=args.cycles,
        schedule_min_esp=args.min_esp,
        schedule_max_esp=args.max_esp,
        num_qubits=args.num_qubits,
        num_runs=args.num_runs
    )
    
    # Example command:
    # python 3.run_nest_single.py --date 03_27 --backend ibm_brussels --schedule optimal --molecule HeH+ --iterations 1 --cycles 6 --num-runs 2
    
    # For direct function call:
    # main(
    #     schedule_type='UP_FLAT',
    #     date='03_27',
    #     backend_name='ibm_brussels',
    #     molecule='H3+',
    #     iterations_per_mapping=72,
    #     n_cycles=6,
    #     schedule_min_esp=1,
    #     schedule_max_esp=1,
    #     num_runs=10  # Run 5 times
    # )
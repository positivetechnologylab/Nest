from utils_mapping import *
from typing import List, Dict, Tuple, Set
import concurrent.futures
from functools import lru_cache
import os
from datetime import datetime

def get_exhausted_subgraph_mapping(n_qubits: int,
                                  adjacency_list: Dict[int, Set[int]],
                                  start_qubit: Optional[int] = None
                                  ) -> Optional[List[int]]:
    """Generate a subgraph mapping starting from a specific qubit"""
    mapping = [start_qubit]
    visited = {start_qubit}
    queue = [start_qubit]
    
    # Track the current spanning tree edges
    spanning_tree_edges = set()
    
    while queue and len(mapping) < n_qubits:
        current = queue.pop(0)
        
        # Sort neighbors by their connectivity to find the most promising paths
        neighbors = list(adjacency_list[current])
        neighbors.sort(key=lambda x: len(adjacency_list[x]), reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                mapping.append(neighbor)
                queue.append(neighbor)
                
                # Add edge to spanning tree
                edge = tuple(sorted([current, neighbor]))
                spanning_tree_edges.add(edge)
                
                if len(mapping) == n_qubits:
                    break
    
    if len(mapping) < n_qubits:
        return None
    
    # Verify the spanning tree is connected and has exactly n_qubits-1 edges
    if len(spanning_tree_edges) != n_qubits - 1:
        return None
    
    return mapping

def spider_find_mappings_only(current_mapping: List[int], 
                           coupling_map: List[List[int]], 
                           used_mappings: Set[frozenset]) -> List[List[int]]:
    """
    Modified spider_find to only generate mappings without calculating ESP.
    Returns a list of mappings only.
    """
    # Pre-compute adjacency maps once for all qubits
    @lru_cache(maxsize=1)
    def build_full_adjacency_map():
        """Build and cache a complete adjacency map for all qubits"""
        adj_map = {}
        for q1, q2 in coupling_map:
            if q1 not in adj_map: adj_map[q1] = set()
            if q2 not in adj_map: adj_map[q2] = set()
            adj_map[q1].add(q2)
            adj_map[q2].add(q1)
        return adj_map
    
    def build_adjacency_map(qubit_map):
        """Build adjacency map for the current mapping"""
        full_adj_map = build_full_adjacency_map()
        adj_map = {q: set() for q in qubit_map}
        for q in qubit_map:
            for neighbor in full_adj_map.get(q, set()):
                if neighbor in qubit_map:
                    adj_map[q].add(neighbor)
        return adj_map
    
    def find_leaf_nodes(adj_map):
        """Find leaf nodes (nodes with only one connection)"""
        return [node for node, neighbors in adj_map.items() if len(neighbors) == 1]
    
    def get_connected_candidates(current_qubits):
        """Get valid candidates that are adjacent to at least one qubit in the current mapping"""
        full_adj_map = build_full_adjacency_map()
        
        candidates = set()
        # Check all qubits not in current mapping
        for qubit in full_adj_map:
            if qubit not in current_qubits:
                # Check if connected to any qubit in current mapping
                if any(neighbor in current_qubits for neighbor in full_adj_map[qubit]):
                    candidates.add(qubit)
        return candidates
    
    print("Starting spider find (mappings only)...")
    mapping_set = set(current_mapping)
    adj_map = build_adjacency_map(mapping_set)
    leaf_nodes = find_leaf_nodes(adj_map)
    
    # Change this to a set for better performance with frozensets
    if not isinstance(used_mappings, set):
        used_mappings = set(frozenset(m) for m in used_mappings)
    
    new_mappings = []
    current_mapping_fs = frozenset(current_mapping)
    
    # Generate all candidate mappings
    for leaf_node in leaf_nodes:
        connected_qubits = get_connected_candidates(mapping_set - {leaf_node})
        
        leaf_idx = current_mapping.index(leaf_node)
        
        for new_qubit in connected_qubits:
            new_mapping = current_mapping.copy()
            new_mapping[leaf_idx] = new_qubit
            new_mapping_fs = frozenset(new_mapping)
            
            if new_mapping_fs != current_mapping_fs and new_mapping_fs not in used_mappings:
                used_mappings.add(new_mapping_fs)  # Mark as used immediately
                new_mappings.append(new_mapping)
    
    if new_mappings:
        print(f"Spider found {len(new_mappings)} new mappings.")
        return new_mappings
    else:
        print(f"No suitable mappings found. Returning empty list.")
        return []

def generate_exhausted_mappings(n_qubits: int, n_mappings: int, coupling_map: List[List[int]], 
                              backend_name: str) -> List[List[int]]:
    """
    Generate mappings without calculating ESP or backend data.
    Returns a list of mappings only.
    """
    mappings_list = []
    used_mappings = set()
    used_qubits = set()
    adjacency_list = build_adjacency_list(coupling_map)
    
    # Create a list of starting points sorted by connectivity
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Building starting points list...")
    
    start_points = [(qubit, len(adjacency_list[qubit])) 
                   for qubit in range(127) if len(adjacency_list[qubit]) >= 2]
    start_points.sort(key=lambda x: x[1], reverse=True)
    start_points = [q for q, _ in start_points]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(start_points)} starting points")
    print(f"Top 10 starting points: {start_points[:10]}")
    
    progress_file = f"{backend_name}_mapping_progress.txt"
    with open(progress_file, 'w') as f:
        f.write(f"Starting exhausted subgraph mapping generation: {datetime.now()}\n")
        f.write(f"Target: {n_mappings} mappings\n\n")
    
    last_update = datetime.now()
    update_interval = 30  # seconds
    total_start_time = datetime.now()
    
    # For each potential starting point
    for idx, start_qubit in enumerate(start_points):
        current_time = datetime.now()
        time_since_update = (current_time - last_update).total_seconds()
        
        # Print progress updates at regular intervals
        if time_since_update >= update_interval or idx % 5 == 0 or idx == 0:
            elapsed = (current_time - total_start_time).total_seconds()
            remaining = (elapsed / (idx+1)) * (len(start_points) - (idx+1)) if idx > 0 else 0
            
            print(f"[{current_time.strftime('%H:%M:%S')}] Processing starting qubit {start_qubit} ({idx + 1}/{len(start_points)})")
            print(f"  Generated {len(mappings_list)}/{n_mappings} mappings ({len(mappings_list)/n_mappings*100:.1f}% complete)")
            print(f"  Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
            last_update = current_time
        
        if len(mappings_list) >= n_mappings:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Generated {len(mappings_list)} mappings, stopping early.")
            break

        temp_adjacency = {k: v.copy() for k, v in adjacency_list.items()}
        mapping = get_exhausted_subgraph_mapping(n_qubits, temp_adjacency, start_qubit)
        
        if mapping is None:
            continue
            
        mapping_tuple = frozenset(mapping)
        if mapping_tuple not in used_mappings:
            try:
                # Add mapping to list
                mappings_list.append(mapping)
                used_mappings.add(mapping_tuple)
                used_qubits.update(mapping)
                
                if len(mappings_list) % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found new mapping from main leg: {mapping}, now total: {len(mappings_list)}")
                
                # Find related mappings using the spider approach (modified to not calculate ESP)
                spider_start = datetime.now()
                related_mappings = spider_find_mappings_only(
                    current_mapping=mapping,
                    coupling_map=coupling_map,
                    used_mappings=used_mappings
                )
                
                if related_mappings:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Spider search found {len(related_mappings)} new mappings in {(datetime.now() - spider_start).total_seconds():.1f}s")
                
                for related_mapping in related_mappings:
                    mappings_list.append(related_mapping)
                
                # Logging and progress updates
                if (datetime.now() - last_update).seconds >= update_interval:
                    progress_file = f"{backend_name}_mapping_progress.txt"
                    with open(progress_file, 'a') as f:
                        f.write(f"Generated {len(mappings_list)} unique mappings: {datetime.now()}\n")
                    
                # Periodic saving of mappings only
                if (datetime.now() - total_start_time).seconds >= 600:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Periodic save after {(datetime.now() - total_start_time).seconds} seconds")
                    mappings_output_file = f"mappings_only_{backend_name}.pkl"
                    logger.info(f"Saving mappings to {mappings_output_file}")
                    with open(mappings_output_file, 'wb') as f:
                        pickle.dump(mappings_list, f)
                    total_start_time = datetime.now()
            
            except Exception as e:
                logger.warning(f"Failed to process mapping {mapping}: {str(e)}")
        else:
            if time_since_update >= update_interval:
                print(f"Mapping {mapping} already used, skipping...")
    
    end_time = datetime.now()
    total_duration = (end_time - total_start_time).total_seconds()
    print(f"[{end_time.strftime('%H:%M:%S')}] Mapping generation complete")
    print(f"  Generated {len(mappings_list)} mappings in {total_duration:.1f} seconds")
    print(f"  Average time per mapping: {total_duration/len(mappings_list):.2f}s")
    
    return mappings_list

def calculate_backend_data(mappings_list: List[List[int]], 
                         circuit: QuantumCircuit, 
                         noise_dir: str, 
                         backend_name: str,
                         num_threads: int = 8) -> List[Dict]:
    """
    Calculate ESP and backend data for the provided mappings using multithreading.
    
    Args:
        mappings_list: List of qubit mappings to process
        circuit: Quantum circuit to use for ESP calculation
        noise_dir: Directory containing noise model data
        backend_name: Name of the quantum backend
        num_threads: Number of parallel threads to use (default: 8)
        
    Returns:
        List of processed mapping data dictionaries
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    mappings_data = []
    total_start_time = datetime.now()
    
    # Thread-safe counters for statistics using threading.Lock
    lock = threading.Lock()
    processed_count = 0
    
    # Statistics tracking
    stats = {
        'total': len(mappings_list),
        'valid': 0,
        'esp_too_low': 0,
        'exception': 0,
        'esp_values': [],
        'processing_times': []
    }
    
    print(f"Calculating backend data for {len(mappings_list)} mappings for backend {backend_name}...")
    print(f"Start time: {total_start_time}")
    print(f"Using {num_threads} parallel threads")
    
    last_update = datetime.now()
    update_interval = 30  # seconds
    
    # Function to process a single mapping in a worker thread
    def process_mapping_task(idx, mapping):
        try:
            mapping_data, status, esp = process_mapping(
                mapping=mapping,
                circuit=circuit,
                noise_dir=noise_dir,
                backend_name=backend_name,
                mapping_id=idx
            )
            
            # Return results to be processed by the main thread
            return {
                'mapping_data': mapping_data,
                'status': status,
                'esp': esp,
                'idx': idx
            }
        except Exception as e:
            # Handle exceptions within the worker thread
            logger.error(f"Error in worker thread processing mapping {idx}: {str(e)}")
            return {
                'mapping_data': None,
                'status': 'exception',
                'esp': 0.0,
                'idx': idx
            }
    
    # Create a thread pool and submit all mapping processing tasks
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_mapping_task, idx, mapping): idx 
            for idx, mapping in enumerate(mappings_list)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_idx):
            result = future.result()
            mapping_data = result['mapping_data']
            status = result['status']
            esp = result['esp']
            idx = result['idx']
            
            # Update shared statistics with thread safety
            with lock:
                processed_count += 1
                stats[status] += 1
                
                if esp > 0:
                    stats['esp_values'].append(esp)
                
                if mapping_data is not None:
                    mappings_data.append(mapping_data)
                    stats['processing_times'].append(mapping_data.get('processing_time', 0))
                
                # Print progress updates at intervals
                current_time = datetime.now()
                time_since_update = (current_time - last_update).total_seconds()
                
                if (time_since_update >= update_interval or 
                    processed_count % 50 == 0 or 
                    processed_count == stats['total']):
                    
                    elapsed = (current_time - total_start_time).total_seconds()
                    remaining = (elapsed / processed_count) * (stats['total'] - processed_count) if processed_count > 0 else 0
                    
                    print(f"[{current_time.strftime('%H:%M:%S')}] Progress: {processed_count}/{stats['total']} mappings ({processed_count/stats['total']*100:.1f}%)")
                    print(f"  Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                    print(f"  Valid: {stats['valid']}, Exceptions: {stats['exception']}")
                    
                    if stats['processing_times']:
                        print(f"  Avg processing time: {np.mean(stats['processing_times']):.2f}s per mapping")
                        print(f"  Thread efficiency: {sum(stats['processing_times'])/elapsed:.2f}x")
                    
                    last_update = current_time
    
    # Calculate the total processing time
    total_time = (datetime.now() - total_start_time).total_seconds()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print(f"BACKEND: {backend_name} PROCESSING SUMMARY")
    print("="*60)
    print(f"Total mappings processed: {stats['total']}")
    print(f"Valid mappings: {stats['valid']} ({stats['valid']/stats['total']*100:.2f}%)")
    print(f"Failed due to exceptions: {stats['exception']} ({stats['exception']/stats['total']*100:.2f}%)")
    
    if stats['esp_values']:
        print(f"ESP statistics:")
        print(f"  Min ESP: {min(stats['esp_values']):.4f}")
        print(f"  Max ESP: {max(stats['esp_values']):.4f}")
        print(f"  Mean ESP: {np.mean(stats['esp_values']):.4f}")
        print(f"  Median ESP: {np.median(stats['esp_values']):.4f}")
    
    if stats['processing_times']:
        print(f"Processing time statistics:")
        print(f"  Min time: {min(stats['processing_times']):.2f} seconds")
        print(f"  Max time: {max(stats['processing_times']):.2f} seconds")
        print(f"  Mean time: {np.mean(stats['processing_times']):.2f} seconds")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per mapping: {total_time/stats['total']:.2f} seconds")
        print(f"  Thread efficiency: {sum(stats['processing_times'])/total_time:.2f}x speedup")
    
    print("="*60 + "\n")
    
    return mappings_data

def process_mapping(mapping: List[int], 
                   circuit: QuantumCircuit, 
                   noise_dir: str, 
                   backend_name: str,
                   mapping_id: int) -> Tuple[Optional[Dict], str, float]:
    start_time = datetime.now()
    try:
        custom_backend = CustomBackend(
            noise_dir=noise_dir,
            qubits_mapping=mapping,
            circuit=circuit,
            backend_name=backend_name
        )
        
        # Calculate ESP for mapping
        esp = custom_backend.calculate_esp(circuit, mapping)
        
        # Create backend data entry - No ESP threshold check
        mapping_data = {
            'mapping': mapping,
            'esp': esp,
            'noise_dir': noise_dir,
            'backend': custom_backend,
            'transpiled_circuit': custom_backend.get_transpiled_circuit(),
            'mapping_id': mapping_id,
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
        
        return mapping_data, 'valid', esp
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing mapping {mapping}: {error_msg}")
        return None, 'exception', 0.0

def save_mappings(mappings_data: List[Dict], output_file: str, ansatz: Optional[QuantumCircuit] = None):
    """Save generated mappings with their metadata to a pickle file."""
    if not mappings_data:
        raise ValueError("No mappings data to save")
        
    data = {
        'mappings_data': mappings_data,
        'metadata': {
            'num_qubits': len(mappings_data[0]['mapping']),
            'num_mappings': len(mappings_data),
            'generation_timestamp': datetime.now().isoformat(),
            'esp_statistics': {
                'min_esp': min(m['esp'] for m in mappings_data),
                'max_esp': max(m['esp'] for m in mappings_data),
                'mean_esp': np.mean([m['esp'] for m in mappings_data]),
                'std_esp': np.std([m['esp'] for m in mappings_data])
            }
        }
    }
    
    if ansatz is not None:
        data['metadata']['ansatz_config'] = {
            'num_qubits': ansatz.num_qubits,
            'reps': 3,
            'type': 'EfficientSU2'
        }
    
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
    
    # Create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved {len(mappings_data)} mappings to {output_file}")

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    date = "03_27"
    num_qubits = 20
    num_mappings = 500
    backend_names = ["ibm_brisbane", "ibm_kyiv", "ibm_brussels", "ibm_sherbrooke", "ibm_strasbourg"]
    num_threads = 8  # Number of threads to use for parallel processing
    
    base_dir = os.path.join(str(num_qubits))
    os.makedirs(base_dir, exist_ok=True)
    
    # Start overall timing
    overall_start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"STARTING EXECUTION: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Date: {date}")
    print(f"  Number of qubits: {num_qubits}")
    print(f"  Target mappings: {num_mappings}")
    print(f"  Backends: {', '.join(backend_names)}")
    print(f"  Threads: {num_threads}")
    print(f"  Output directory: {base_dir}")
    print(f"{'='*80}\n")
    
    ansatz = EfficientSU2(num_qubits=num_qubits, reps=3)
    
    # Step 1: Generate mappings once (using coupling map from the first backend)
    # Since all backends have the same coupling map, we only need to generate mappings once
    first_backend = backend_names[0]
    noise_dir = f"noise_models/{date}/{first_backend}"
    
    # Time loading of noise data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading noise data from {noise_dir}...")
    load_start = datetime.now()
    noise_model, coupling_map, basis_gates, _, _ = load_noise_data(noise_dir)
    load_time = (datetime.now() - load_start).total_seconds()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded noise data in {load_time:.2f} seconds")
    
    # Time mapping generation
    mapping_start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"GENERATING MAPPINGS: {mapping_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Generate mappings only, without calculating ESP
    mappings_list = generate_exhausted_mappings(num_qubits, num_mappings, coupling_map, "common")
    
    mapping_end_time = datetime.now()
    mapping_duration = mapping_end_time - mapping_start_time
    print(f"\n{'='*80}")
    print(f"MAPPING GENERATION COMPLETE: {mapping_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total mappings: {len(mappings_list)}")
    print(f"Time taken: {mapping_duration}")
    print(f"Average time per mapping: {mapping_duration.total_seconds() / max(1, len(mappings_list)):.2f} seconds")
    print(f"{'='*80}\n")
    
    # Save the mappings list (without ESP data) for potential recovery
    save_start = datetime.now()
    print(f"[{save_start.strftime('%H:%M:%S')}] Saving common mappings...")
    mappings_output_file = os.path.join(base_dir, "common_mappings.pkl")
    with open(mappings_output_file, 'wb') as f:
        pickle.dump(mappings_list, f)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved mappings in {(datetime.now() - save_start).total_seconds():.2f} seconds")
    
    # Step 2: Calculate ESP and backend data for each backend using the same mappings
    for backend_idx, backend_name in enumerate(backend_names):
        backend_start_time = datetime.now()
        print(f"\n{'='*80}")
        print(f"BACKEND {backend_idx+1}/{len(backend_names)}: {backend_name}")
        print(f"Starting ESP calculation: {backend_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {num_threads} threads for parallel processing")
        print(f"{'='*80}")
        
        noise_dir = f"noise_models/{date}/{backend_name}"
        
        # Calculate ESP and backend data for each mapping using multithreading
        mappings_data = calculate_backend_data(
            mappings_list=mappings_list,
            circuit=ansatz,
            noise_dir=noise_dir,
            backend_name=backend_name,
            num_threads=num_threads
        )
        
        # Create backend-specific directory
        backend_dir = os.path.join(base_dir, backend_name)
        os.makedirs(backend_dir, exist_ok=True)
        
        # Save mappings with new path structure
        output_file = os.path.join(backend_dir, "mappings.pkl")
        logger.info(f"Saving data to {output_file}")
        save_start = datetime.now()
        print(f"[{save_start.strftime('%H:%M:%S')}] Saving backend data to {output_file}...")
        save_mappings(mappings_data=mappings_data, output_file=output_file, ansatz=ansatz)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved data in {(datetime.now() - save_start).total_seconds():.2f} seconds")
        
        backend_end_time = datetime.now()
        backend_duration = backend_end_time - backend_start_time
        print(f"\n{'='*80}")
        print(f"BACKEND {backend_name} COMPLETE")
        print(f"Start: {backend_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End: {backend_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {backend_duration}")
        print(f"Time per mapping: {backend_duration.total_seconds() / len(mappings_list):.2f} seconds")
        print(f"{'='*80}\n")
    
    # Print overall timing summary
    overall_end_time = datetime.now()
    overall_duration = overall_end_time - overall_start_time
    print(f"\n{'='*80}")
    print(f"EXECUTION COMPLETE")
    print(f"Started: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {overall_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {overall_duration}")
    print(f"{'='*80}")
    
    # Print backend processing times if multiple backends
    if len(backend_names) > 1:
        print("\nTime summary per backend:")
        for backend_name in backend_names:
            print(f"  - {backend_name}: {backend_duration}")
    
    print(f"\nTotal number of mappings: {len(mappings_list)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
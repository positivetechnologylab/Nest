import numpy as np
import logging
import os
import pickle
import json
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2

logger = logging.getLogger(__name__)

class CustomBackend:
    """Custom quantum backend using a subset of qubits from a larger backend"""
    def __init__(self, noise_dir: str, qubits_mapping: List[int], circuit: QuantumCircuit, backend_name: str):
        self.backend_name = backend_name
        self.noise_model, self.coupling_map, self.basis_gates, t1_times, t2_times = self._load_noise_data(noise_dir)
        self.qubits_mapping = qubits_mapping
        self.num_qubits = len(qubits_mapping)
        self.qubit_index_map = {orig: idx for idx, orig in enumerate(qubits_mapping)}
        
        self.noise_model = self._filter_noise_model(self.noise_model, qubits_mapping)
        self.coupling_map = self._filter_coupling_map(self.coupling_map, qubits_mapping)
        
        self.t1_times = {qubits_mapping.index(k): v for k, v in t1_times.items() if k in qubits_mapping}
        self.t2_times = {qubits_mapping.index(k): v for k, v in t2_times.items() if k in qubits_mapping}
        
        self.simulator = AerSimulator(
            noise_model=self.noise_model,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            shots=1024
        )
        self.simulator.set_options(
            device='GPU'
        )
        
        self.transpiled_circuit = self._get_transpiled_circuit(circuit)

    def _load_noise_data(self, noise_dir):
        return load_noise_data(noise_dir)
        
    def _filter_noise_model(self, noise_model, qubit_mapping):
        filtered_model = NoiseModel(noise_model.basis_gates)
        
        all_qubits = set(range(max(noise_model._local_quantum_errors.get('sx', {}).keys(), key=lambda x: x[0])[0] + 1))
        qubits_to_keep = set(qubit_mapping)
        
        for gate, error_dict in noise_model._local_quantum_errors.items():
            if isinstance(error_dict, dict):
                for error_key, error in error_dict.items():
                    if len(error_key) == 1:
                        if error_key[0] in qubits_to_keep:
                            filtered_model.add_quantum_error(
                                error,
                                gate,
                                [self.qubit_index_map[error_key[0]]]
                            )
                    elif len(error_key) == 2:
                        if error_key[0] in qubits_to_keep and error_key[1] in qubits_to_keep:
                            filtered_model.add_quantum_error(
                                error,
                                gate,
                                [self.qubit_index_map[error_key[0]], 
                                self.qubit_index_map[error_key[1]]]
                            )
        
        return filtered_model

    def _filter_coupling_map(self, coupling_map, qubit_mapping):
        filtered_map = []
        for pair in coupling_map:
            if pair[0] in qubit_mapping and pair[1] in qubit_mapping:
                new_pair = [
                    self.qubit_index_map[pair[0]],
                    self.qubit_index_map[pair[1]]
                ]
                filtered_map.append(new_pair)
        return filtered_map

    def _get_transpiled_circuit(self, circuit):
        adjusted_layout = [self.qubit_index_map[pos] for pos in self.qubits_mapping]
        try:
            transpiled_circuit = transpile(
                circuit,
                backend=self.simulator,
                initial_layout=adjusted_layout,
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates,
                optimization_level=3
            )
            return transpiled_circuit

        except Exception as e:
            logger.error(f"Error in transpilation: {str(e)}")
            raise

    def calculate_esp(self, circuit, initial_layout):
        try:
            transpiled_circuit = self.transpiled_circuit
            circuit_depth = transpiled_circuit.depth()
            esp = 1.0
            
            for gate in transpiled_circuit.data:
                gate_operation = gate.operation
                gate_name = gate_operation.name
                gate_qubits = [transpiled_circuit.find_bit(q).index for q in gate.qubits]
        
                if gate_name in self.noise_model.noise_instructions:
                    error = None
                    qubit_tuple = tuple(gate_qubits)
                    
                    if gate_name in self.noise_model._local_quantum_errors:
                        gate_errors = self.noise_model._local_quantum_errors[gate_name]
                        if qubit_tuple in gate_errors:
                            error = gate_errors[qubit_tuple]
                    
                    if error is not None:
                        try:
                            success_probability = error.probabilities[0]
                            esp *= success_probability
                        except Exception as e:
                            logger.warning(f"Error processing gate {gate_name} on qubits {gate_qubits}: {str(e)}")
                            continue
            
            # Calculate decoherence effects using average T1 and T2 times
            avg_t1 = np.mean(list(self.t1_times.values()))
            avg_t2 = np.mean(list(self.t2_times.values()))
            gate_time = 50e-9  # Approximate gate time in seconds
            total_time = circuit_depth * gate_time
            
            decoherence = np.exp(-total_time / avg_t1) * np.exp(-total_time / avg_t2)
            esp *= decoherence
            
            return max(0.0001, esp)
        except Exception as e:
            logger.error(f"Error calculating ESP: {str(e)}")
            raise

    def get_transpiled_circuit(self) -> QuantumCircuit:
        return self.transpiled_circuit
    
    def get_backend_parameters(self) -> Tuple[NoiseModel, List, List]:
        return self.noise_model, self.coupling_map, self.basis_gates
    
    def get_qubit_index_map(self) -> Dict[int, int]:
        return self.qubit_index_map
    
def load_noise_data(noise_dir: str) -> Tuple[NoiseModel, List, List, Dict[int, float], Dict[int, float]]:
    """Load noise model and properties from files."""
    try:
        noise_model_path = os.path.join(noise_dir, 'noise_model.pkl')
        if not os.path.exists(noise_model_path):
            raise FileNotFoundError(f"Noise model file not found at {noise_model_path}")
            
        with open(noise_model_path, 'rb') as f:
            noise_model = pickle.load(f)
        
        properties_path = os.path.join(noise_dir, 'properties.json')
        if not os.path.exists(properties_path):
            raise FileNotFoundError(f"Properties file not found at {properties_path}")
            
        with open(properties_path, 'r') as f:
            properties = json.load(f)
        
        relaxation_times = properties.get('relaxation_times', {})
        t1_times = {int(k): float(v) for k, v in relaxation_times.get('t1_times', {}).items()}
        t2_times = {int(k): float(v) for k, v in relaxation_times.get('t2_times', {}).items()}
        
        return noise_model, properties['coupling_map'], properties['basis_gates'], t1_times, t2_times
            
    except Exception as e:
        logger.error(f"Error loading noise data: {str(e)}")
        raise
    
def build_adjacency_list(coupling_map: List[List[int]], max_qubit: int = 127) -> Dict[int, Set[int]]:
    """Build an adjacency list from the coupling map for faster lookups."""
    adjacency = {i: set() for i in range(max_qubit + 1)}
    for edge in coupling_map:
        adjacency[edge[0]].add(edge[1])
        adjacency[edge[1]].add(edge[0])
    return adjacency


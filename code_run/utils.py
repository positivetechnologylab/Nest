from pennylane import numpy as np
import logging
import os
import pickle
import json
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import transpile
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
from datetime import datetime
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_transpiler_service.transpiler_service import TranspilerService
from typing import List, Dict, Tuple
import pennylane as qml

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import threading
plot_lock = threading.Lock()

last_mapping_by_schedule = {}
iterations_per_mapping = 36
n_cycles = 12

logger = logging.getLogger(__name__)

####################################
######## Plotting Functions ########
####################################

def plot_dual_mapping_results(results, output_dir):
    """Create plots comparing results from dual mappings with different schedules."""
    os.makedirs(output_dir, exist_ok=True)
    
    schedule_type1 = results['schedule_type1']
    schedule_type2 = results['schedule_type2']
    
    # Plot 1: Energy Convergence for Both Mappings
    with plot_lock:
        plt.figure(figsize=(12, 6))
        
        energies1 = results['vqe_runner1'].optimization_history['energies']
        energies2 = results['vqe_runner2'].optimization_history['energies']
        
        iters1 = range(len(energies1))
        iters2 = range(len(energies2))
        
        plt.plot(iters1, energies1, 'b-', label=f"Mapping 1 ({schedule_type1})", linewidth=2)
        plt.plot(iters2, energies2, 'r-', label=f"Mapping 2 ({schedule_type2})", linewidth=2)
        
        # Mark the cycle boundaries with vertical lines
        cycle_boundaries = np.cumsum([iterations_per_mapping] * (n_cycles-1))
        for boundary in cycle_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f"Energy Convergence for Dual Mappings ({schedule_type1} vs {schedule_type2})")
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dual_mapping_energy_{schedule_type1}_vs_{schedule_type2}.png"), dpi=300)
        plt.close()
    
    # Plot 2: ESP Schedules
    with plot_lock:
        plt.figure(figsize=(12, 6))
        
        # Target ESP schedules
        cycles = range(1, n_cycles + 1)
        target_esp1 = results['esp_schedule1']
        target_esp2 = results['esp_schedule2']
        
        # Actual ESP schedules
        actual_esp1 = results['mapping1_history']['actual_esp_schedule']
        actual_esp2 = results['mapping2_history']['actual_esp_schedule']
        
        plt.plot(cycles, target_esp1, 'b--', label=f"Target ESP ({schedule_type1})", alpha=0.7)
        plt.plot(cycles, actual_esp1, 'b-', label=f"Actual ESP ({schedule_type1})", linewidth=2)
        plt.plot(cycles, target_esp2, 'r--', label=f"Target ESP ({schedule_type2})", alpha=0.7)
        plt.plot(cycles, actual_esp2, 'r-', label=f"Actual ESP ({schedule_type2})", linewidth=2)
        
        plt.title(f"ESP Schedules for Dual Mappings ({schedule_type1} vs {schedule_type2})")
        plt.xlabel('Cycle')
        plt.ylabel('ESP Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dual_mapping_esp_{schedule_type1}_vs_{schedule_type2}.png"), dpi=300)
        plt.close()
    
    print(f"Plots saved to {output_dir} directory")

def save_dual_mapping_results(results, output_dir):
    """Save dual-mapping experimental results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Create serializable results
    serializable_results = {
        'schedule_type1': results['schedule_type1'],
        'schedule_type2': results['schedule_type2'],
        'run_time': convert_to_native(results.get('run_time', 0)),
        'mapping1': {
            'target_esp_schedule': [convert_to_native(x) for x in results['mapping1_history']['target_esp_schedule']],
            'actual_esp_schedule': [convert_to_native(x) for x in results['mapping1_history']['actual_esp_schedule']],
            'selected_mappings': [[convert_to_native(x) for x in m] for m in results['mapping1_history']['selected_mappings']],
            'energies': [convert_to_native(x) for x in results['vqe_runner1'].optimization_history['energies']],
            'final_results': [
                {
                    'fun': convert_to_native(r.fun),
                    'success': bool(r.success),
                    'message': str(r.message)
                } for r in results['mapping1_results']
            ]
        },
        'mapping2': {
            'target_esp_schedule': [convert_to_native(x) for x in results['mapping2_history']['target_esp_schedule']],
            'actual_esp_schedule': [convert_to_native(x) for x in results['mapping2_history']['actual_esp_schedule']],
            'selected_mappings': [[convert_to_native(x) for x in m] for m in results['mapping2_history']['selected_mappings']],
            'energies': [convert_to_native(x) for x in results['vqe_runner2'].optimization_history['energies']],
            'final_results': [
                {
                    'fun': convert_to_native(r.fun),
                    'success': bool(r.success),
                    'message': str(r.message)
                } for r in results['mapping2_results']
            ]
        }
    }
    
    filename = os.path.join(
        output_dir,
        f'dual_mapping_{results["schedule_type1"]}_vs_{results["schedule_type2"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return filename

####################################
########## CustomBackend ###########
####################################

class CustomBackend:
    """Custom quantum backend using a subset of qubits from a larger backend"""
    def __init__(self, noise_dir: str, qubits_mapping: List[int], circuit: QuantumCircuit, backend_name: str):
        self.backend_name = backend_name
        self.noise_model, self.coupling_map, self.basis_gates, t1_times, t2_times, properties = self._load_noise_data(noise_dir)

        self.qubits_mapping = qubits_mapping
        self.num_qubits = len(qubits_mapping)
        self.qubit_index_map = {orig: idx for idx, orig in enumerate(qubits_mapping)}
        print(f"Qubit index map: {self.qubit_index_map}")
        
        self.noise_model = self._filter_noise_model(self.noise_model, qubits_mapping)
        self.coupling_map = self._filter_coupling_map(self.coupling_map, qubits_mapping)
        self.properties = self._filter_properties(properties)
        
        self.t1_times = {qubits_mapping.index(k): v for k, v in t1_times.items() if k in qubits_mapping}
        self.t2_times = {qubits_mapping.index(k): v for k, v in t2_times.items() if k in qubits_mapping}
        
        self.simulator = AerSimulator(
            noise_model=self.noise_model,
            coupling_map=self.coupling_map,
            basis_gates=self.basis_gates,
            shots=1024
        )
        self.simulator.set_options(device="GPU")
        
        self.transpiled_circuit = self._get_transpiled_circuit(circuit)

    def _load_noise_data(self, noise_dir: str):
        """Load noise model and properties from files"""
        try:
            # Load noise model
            with open(os.path.join(noise_dir, 'noise_model.pkl'), 'rb') as f:
                noise_model = pickle.load(f)
            
            # Load properties
            with open(os.path.join(noise_dir, 'properties.json'), 'r') as f:
                properties = json.load(f)
            
            # Extract relaxation times
            relaxation_times = properties.get('relaxation_times', {})
            t1_times = {int(k): float(v) for k, v in relaxation_times.get('t1_times', {}).items()}
            t2_times = {int(k): float(v) for k, v in relaxation_times.get('t2_times', {}).items()}
            
            return noise_model, properties['coupling_map'], properties['basis_gates'], t1_times, t2_times, properties
            
        except Exception as e:
            logger.error(f"Error loading noise data: {str(e)}")
            raise
    
    def _filter_noise_model(self, noise_model: NoiseModel, qubit_mapping: List[int]) -> NoiseModel:
        # Create new noise model with same basis gates
        filtered_model = NoiseModel(noise_model.basis_gates)
        
        # Get all qubits in the model
        all_qubits = set(range(max(noise_model._local_quantum_errors.get('sx', {}).keys(), key=lambda x: x[0])[0] + 1))
        qubits_to_keep = set(qubit_mapping)
        
        # Filter and copy quantum errors
        for gate, error_dict in noise_model._local_quantum_errors.items():
            if isinstance(error_dict, dict):
                for error_key, error in error_dict.items():
                    # Single qubit gates
                    if len(error_key) == 1:
                        if error_key[0] in qubits_to_keep:
                            filtered_model.add_quantum_error(
                                error,
                                gate,
                                [self.qubit_index_map[error_key[0]]]
                            )
                    # Two qubit gates
                    elif len(error_key) == 2:
                        if error_key[0] in qubits_to_keep and error_key[1] in qubits_to_keep:
                            filtered_model.add_quantum_error(
                                error,
                                gate,
                                [self.qubit_index_map[error_key[0]], 
                                self.qubit_index_map[error_key[1]]]
                            )
            else:
                # Handle non-dict error case (if any)
                for qubit in qubits_to_keep:
                    filtered_model.add_quantum_error(
                        error_dict,
                        gate,
                        [self.qubit_index_map[qubit]]
                    )
        
        # Filter and copy reset errors
        if hasattr(noise_model, '_local_reset_errors'):
            for qubit, error in noise_model._local_reset_errors.items():
                if qubit in qubits_to_keep:
                    filtered_model.add_quantum_error(
                        error,
                        'reset',
                        [self.qubit_index_map[qubit]]
                    )
        
        # Filter and copy readout errors
        if hasattr(noise_model, '_local_readout_errors'):
            for qubit, error in noise_model._local_readout_errors.items():
                if qubit in qubits_to_keep:
                    filtered_model.add_readout_error(
                        error,
                        [self.qubit_index_map[qubit]]
                    )
        
        return filtered_model
    
    def _filter_coupling_map(self, coupling_map: List[List[int]], qubit_mapping: List[int]) -> List[List[int]]:
        """Filter coupling map to only include specified qubits"""
        filtered_map = []
        for pair in coupling_map:
            if pair[0] in qubit_mapping and pair[1] in qubit_mapping:
                new_pair = [
                    self.qubit_index_map[pair[0]],
                    self.qubit_index_map[pair[1]]
                ]
                filtered_map.append(new_pair)
        return filtered_map
    
    def _filter_properties(self, properties: Dict) -> Dict:
        filtered_properties = {}
        
        # Handle relaxation times
        if 'relaxation_times' in properties:
            relaxation_times = properties['relaxation_times']
            filtered_relaxation = {}
            
            # Filter and remap T1 times
            if 't1_times' in relaxation_times:
                filtered_t1 = {}
                for orig_qubit, value in relaxation_times['t1_times'].items():
                    if int(orig_qubit) in self.qubits_mapping:
                        new_qubit = self.qubit_index_map[int(orig_qubit)]
                        filtered_t1[str(new_qubit)] = value
                filtered_relaxation['t1_times'] = filtered_t1
            
            # Filter and remap T2 times
            if 't2_times' in relaxation_times:
                filtered_t2 = {}
                for orig_qubit, value in relaxation_times['t2_times'].items():
                    if int(orig_qubit) in self.qubits_mapping:
                        new_qubit = self.qubit_index_map[int(orig_qubit)]
                        filtered_t2[str(new_qubit)] = value
                filtered_relaxation['t2_times'] = filtered_t2
            
            filtered_properties['relaxation_times'] = filtered_relaxation
        
        # Handle measurement errors
        if 'measurement_errors' in properties:
            filtered_meas = {}
            for orig_qubit, value in properties['measurement_errors'].items():
                if int(orig_qubit) in self.qubits_mapping:
                    new_qubit = self.qubit_index_map[int(orig_qubit)]
                    filtered_meas[str(new_qubit)] = value
            filtered_properties['measurement_errors'] = filtered_meas
        
        # Handle other properties that might be qubit-specific
        for key, value in properties.items():
            if key not in ['relaxation_times', 'measurement_errors']:
                if isinstance(value, dict):
                    filtered_value = {}
                    for orig_qubit, qubit_value in value.items():
                        if int(orig_qubit) in self.qubits_mapping:
                            new_qubit = self.qubit_index_map[int(orig_qubit)]
                            filtered_value[str(new_qubit)] = qubit_value
                    filtered_properties[key] = filtered_value
                else:
                    filtered_properties[key] = value
        
        return filtered_properties
    
    def _get_transpiled_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
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
        except Exception as e:
            print(f"Error transpile: {str(e)}")
        return transpiled_circuit
        
    def get_transpiled_circuit(self) -> QuantumCircuit:
        return self.transpiled_circuit
    
    def get_backend(self) -> AerSimulator:
        return self.simulator
    
    def get_backend_parameters(self) -> Tuple[NoiseModel, List, List]:
        return self.noise_model, self.coupling_map, self.basis_gates
    
    def get_qubit_index_map(self) -> Dict[int, int]:
        return self.qubit_index_map
    
    def get_properties(self) -> Dict:
        return self.properties

    def get_noise_model(self) -> NoiseModel:
        return self.noise_model
    
    def get_coupling_map(self) -> List[List[int]]:
        return self.coupling_map
    
    def get_basis_gates(self) -> List[str]:
        return self.basis_gates
    
    def calculate_esp(self, circuit, initial_layout) -> float:
        adjusted_layout = [self.qubit_index_map[pos] for pos in initial_layout]
        try:
            transpiled_circuit = self.transpiled_circuit
            
            circuit_depth = transpiled_circuit.depth()
            esp = 1.0  # Initialize ESP
            
            # Iterate over all gates in the transpiled circuit
            for gate in transpiled_circuit.data:
                gate_operation = gate.operation
                gate_name = gate_operation.name
                gate_qubits = [transpiled_circuit.find_bit(q).index for q in gate.qubits]
        
                if gate_name in self.noise_model.noise_instructions:
                    error = None
                    qubit_tuple = tuple(gate_qubits)
                    
                    # The local quantum errors are organized by gate_name first
                    if gate_name in self.noise_model._local_quantum_errors:
                        gate_errors = self.noise_model._local_quantum_errors[gate_name]
                        
                        # Look for error for these specific qubits
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
            
            # Clip ESP to 0 if it's too small
            if esp < 0.0001:
                esp = 0
                
            return esp
        except Exception as e:
            logger.error(f"Error calculating ESP: {str(e)}")
            raise
    
class ESPScheduler:
    """Implements ESP scheduling for VQE optimization."""
    def __init__(self, min_value: float, max_value: float, n_cycles: int):
        self.min_value = min_value
        self.max_value = max_value
        self.n_cycles = n_cycles
        logger.info(f"Initialized ESPScheduler with min={min_value}, max={max_value}, cycles={n_cycles}")
    
    def get_LR_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles // 2
        decrease = []
        for i in range(n_levels):
            stepped_pos = i / (n_levels - 1)
            value = max_esp - (max_esp - min_esp) * stepped_pos
            decrease.append(value)
        increase = decrease[::-1]
        return increase + increase

    def get_CR_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles // 2
        t = np.linspace(3*np.pi/2, 2*np.pi, n_levels)
        wave = np.cos(t)
        section = (min_esp + wave * (max_esp - min_esp)).tolist()
        return section + section

    def get_LT_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles // 2
        decrease = []
        for i in range(n_levels):
            stepped_pos = i / (n_levels - 1)
            value = max_esp - (max_esp - min_esp) * stepped_pos
            decrease.append(value)
        increase = decrease[::-1]
        return decrease + increase
    
    def get_LTT_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles // 2
        decrease = []
        for i in range(n_levels):
            stepped_pos = i / (n_levels - 1)
            value = max_esp - (max_esp - min_esp) * stepped_pos
            decrease.append(value)
        increase = decrease[::-1]
        return increase + decrease

    def get_CT_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles
        t = np.linspace(0, 2*np.pi, n_levels)
        wave = np.cos(t)
        return (min_esp + (wave - (-1)) * (max_esp - min_esp) / (1 - (-1))).tolist()
    
    def get_CTT_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles
        t = np.linspace(np.pi, 3*np.pi, n_levels)
        wave = np.cos(t)
        return (min_esp + (wave - (-1)) * (max_esp - min_esp) / (1 - (-1))).tolist()
    
    def get_RR_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_steps = n_cycles // 2 
        schedule = []
        for _ in range(2):
            x = np.linspace(0, 1, n_steps)
            increase = np.exp(x * np.log(max_esp/min_esp)) * min_esp
            schedule.extend(increase.tolist())
        return schedule

    def get_ER_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_steps = n_cycles // 2
        x = np.linspace(0, 1, n_steps)
        increase = np.exp(x * np.log(max_esp / min_esp)) * min_esp
        shifted_flipped = max_esp - (increase - min_esp)
        horizontal_flipped = shifted_flipped[::-1]
        return horizontal_flipped.tolist() * 2

    def get_RTV_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_steps = n_cycles // 2
        x = np.linspace(0, 1, n_steps)
        increase = np.exp(x * np.log(max_esp / min_esp)) * min_esp
        shifted_flipped = max_esp - (increase - min_esp)
        return shifted_flipped.tolist() + increase.tolist()

    def get_RTH_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_steps = n_cycles // 2
        x = np.linspace(0, 1, n_steps)
        decrease = max_esp / np.exp(x * np.log(max_esp / min_esp))
        increase = np.exp(x * np.log(max_esp/min_esp)) * min_esp
        return decrease.tolist() + increase.tolist()

    def get_ETV_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_steps = n_cycles // 2
        x = np.linspace(0, 1, n_steps)
        decrease = max_esp / np.exp(x * np.log(max_esp / min_esp))
        increase = np.exp(x * np.log(max_esp / min_esp)) * min_esp
        shifted_flipped = max_esp - (increase - min_esp)
        return decrease.tolist() + shifted_flipped[::-1].tolist()

    def get_ETH_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_steps = n_cycles // 2
        x = np.linspace(0, 1, n_steps)
        increase = np.exp(x * np.log(max_esp / min_esp)) * min_esp
        shifted_flipped = max_esp - (increase - min_esp)
        return shifted_flipped.tolist() + shifted_flipped[::-1].tolist()
    
    # NEW SINGLE PATTERN SCHEDULES
    
    def get_LR_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles
        decrease = []
        for i in range(n_levels):
            stepped_pos = i / (n_levels - 1)
            value = max_esp - (max_esp - min_esp) * stepped_pos
            decrease.append(value)
        increase = decrease[::-1]
        return increase

    def get_CR_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Cosine Ramp - single cosine wave over all cycles
        min_esp, max_esp = esp_range
        t = np.linspace(3*np.pi/2, 7*np.pi/2, n_cycles)  # Complete cosine cycle
        wave = np.cos(t)
        return (min_esp + (wave - (-1)) * (max_esp - min_esp) / 2).tolist()

    def get_LT_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Linear Triangle - single triangle over all cycles
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            stepped_pos = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # Descending part
                value = max_esp - (max_esp - min_esp) * (stepped_pos * 2)
            else:
                # Ascending part
                mid_pos = (i - n_cycles // 2) / (n_cycles // 2)
                value = min_esp + (max_esp - min_esp) * mid_pos
            values.append(value)
        return values
    
    def get_LTT_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Linear Triangle (inverted) - single inverted triangle over all cycles
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            stepped_pos = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # Ascending part
                value = min_esp + (max_esp - min_esp) * (stepped_pos * 2)
            else:
                # Descending part
                mid_pos = (i - n_cycles // 2) / (n_cycles // 2)
                value = max_esp - (max_esp - min_esp) * mid_pos
            values.append(value)
        return values

    def get_CT_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Cosine Triangle - single cosine cycle over all cycles (already single in original)
        return self.get_CT_schedule(n_cycles, esp_range)
    
    def get_CTT_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Cosine Triangle (inverted) - single inverted cosine cycle (already single in original)
        return self.get_CTT_schedule(n_cycles, esp_range)
    
    def get_RR_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Repeated Ramp - one exponential ramp up and down
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            x = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # First half: exponential increase
                half_x = x * 2  # Rescale to [0,1]
                value = np.exp(half_x * np.log(max_esp/min_esp)) * min_esp
            else:
                # Second half: exponential increase again
                half_x = (i - n_cycles // 2) / (n_cycles // 2)  # Rescale to [0,1]
                value = np.exp(half_x * np.log(max_esp/min_esp)) * min_esp
            values.append(value)
        return values

    def get_ER_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Exponential Ramp
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            x = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # First half: exponential decrease
                half_x = x * 2  # Rescale to [0,1]
                exp_value = np.exp(half_x * np.log(max_esp / min_esp)) * min_esp
                value = max_esp - (exp_value - min_esp)
            else:
                # Second half: exponential decrease (repeated)
                half_x = (i - n_cycles // 2) / (n_cycles // 2)  # Rescale to [0,1]
                exp_value = np.exp(half_x * np.log(max_esp / min_esp)) * min_esp
                value = max_esp - (exp_value - min_esp)
            values.append(value)
        return values

    def get_RTV_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Ramp Triangle V-shaped - single pattern
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            x = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # First half: exponential decrease
                half_x = x * 2  # Rescale to [0,1]
                exp_value = np.exp(half_x * np.log(max_esp / min_esp)) * min_esp
                value = max_esp - (exp_value - min_esp)
            else:
                # Second half: exponential increase
                half_x = (i - n_cycles // 2) / (n_cycles // 2)  # Rescale to [0,1]
                value = np.exp(half_x * np.log(max_esp/min_esp)) * min_esp
            values.append(value)
        return values

    def get_RTH_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Ramp Triangle H-shaped - single pattern
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            x = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # First half: exponential decrease
                half_x = x * 2  # Rescale to [0,1]
                value = max_esp / np.exp(half_x * np.log(max_esp / min_esp))
            else:
                # Second half: exponential increase
                half_x = (i - n_cycles // 2) / (n_cycles // 2)  # Rescale to [0,1]
                value = np.exp(half_x * np.log(max_esp/min_esp)) * min_esp
            values.append(value)
        return values

    def get_ETV_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Exponential Triangle V-shaped - single pattern
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            x = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # First half: exponential decrease
                half_x = x * 2  # Rescale to [0,1]
                value = max_esp / np.exp(half_x * np.log(max_esp / min_esp))
            else:
                # Second half: flipped exponential increase (reversed)
                half_x = (i - n_cycles // 2) / (n_cycles // 2)  # Rescale to [0,1]
                exp_value = np.exp((1-half_x) * np.log(max_esp / min_esp)) * min_esp
                value = max_esp - (exp_value - min_esp)
            values.append(value)
        return values

    def get_ETH_single_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        # Exponential Triangle H-shaped - single pattern
        min_esp, max_esp = esp_range
        values = []
        for i in range(n_cycles):
            x = i / (n_cycles - 1)
            if i < n_cycles // 2:
                # First half: flipped exponential increase
                half_x = x * 2  # Rescale to [0,1]
                exp_value = np.exp(half_x * np.log(max_esp / min_esp)) * min_esp
                value = max_esp - (exp_value - min_esp)
            else:
                # Second half: flipped exponential decrease
                half_x = (i - n_cycles // 2) / (n_cycles // 2)  # Rescale to [0,1]
                exp_value = np.exp((1-half_x) * np.log(max_esp / min_esp)) * min_esp
                value = max_esp - (exp_value - min_esp)
            values.append(value)
        return values
    
    def get_UP_FLAT_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        n_levels = n_cycles // 2
        
        if n_levels == 1:  # Special case for n_cycles = 2
            return [min_esp, max_esp]
        
        decrease = []
        for i in range(n_levels):
            stepped_pos = i / (n_levels - 1)  # Now safe since n_levels > 1
            value = max_esp - (max_esp - min_esp) * stepped_pos
            decrease.append(value)
        
        increase = decrease[::-1]  # Reverse to get increasing values
        flat = [max_esp] * n_levels  # Create flat section at max_esp
        
        return increase + flat
    
    def get_RELU_schedule(self, n_cycles: int, esp_range: Tuple[float, float]) -> List[float]:
        min_esp, max_esp = esp_range
        half_cycles = n_cycles // 2
        
        # First half: flat at minimum value
        flat_part = [min_esp] * half_cycles
        
        # Second half: linear increase from min to max
        increase_part = []
        for i in range(half_cycles):
            if half_cycles == 1:  # Handle edge case of only one step in the increase
                stepped_pos = 1.0
            else:
                stepped_pos = i / (half_cycles - 1)
            value = min_esp + (max_esp - min_esp) * stepped_pos
            increase_part.append(value)
        
        return flat_part + increase_part
    
    def get_schedule(self, n_cycles: int, esp_range: Tuple[float, float], schedule_type: str = 'CR') -> List[float]:
        schedule_map = {
            'LR': self.get_LR_schedule,
            'CR': self.get_CR_schedule,
            'LT': self.get_LT_schedule,
            'CT': self.get_CT_schedule,
            'RR': self.get_RR_schedule,
            'ER': self.get_ER_schedule,
            'RTV': self.get_RTV_schedule, 
            'RTH': self.get_RTH_schedule,
            'ETV': self.get_ETV_schedule,
            'ETH': self.get_ETH_schedule,
            'LTT': self.get_LTT_schedule,
            'CTT': self.get_CTT_schedule,
            'LR_single': self.get_LR_single_schedule,
            'CR_single': self.get_CR_single_schedule,
            'LT_single': self.get_LT_single_schedule,
            'CT_single': self.get_CT_single_schedule,
            'RR_single': self.get_RR_single_schedule,
            'ER_single': self.get_ER_single_schedule,
            'RTV_single': self.get_RTV_single_schedule, 
            'RTH_single': self.get_RTH_single_schedule,
            'ETV_single': self.get_ETV_single_schedule,
            'ETH_single': self.get_ETH_single_schedule,
            'LTT_single': self.get_LTT_single_schedule,
            'CTT_single': self.get_CTT_single_schedule,
            'UP_FLAT': self.get_UP_FLAT_schedule,
            'RELU': self.get_RELU_schedule
        }
        
        if schedule_type not in schedule_map:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        return schedule_map[schedule_type](n_cycles, esp_range)

####################################
############ VQERunner #############
####################################

class VQERunner:
    def __init__(self, hamiltonian: SparsePauliOp, ansatz: QuantumCircuit, custom_backend: Dict, shots: int = 1024):
        self.original_hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.shots = shots
        
        # Extract backend parameters
        self.transpiled_circuit = custom_backend.get_transpiled_circuit()
        self.backend_instance = custom_backend
        
        simulator = AerSimulator(
            method="density_matrix",
            noise_model=self.backend_instance.noise_model,
            coupling_map=self.backend_instance.coupling_map,
            basis_gates=self.backend_instance.basis_gates,
            shots=self.shots
        )
        simulator.set_options(device="GPU")
        
        self.estimator = BackendEstimatorV2(backend=simulator)
        
        self.optimization_history = {
            'energies': [],
            'parameters': [],
            'iterations': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_expectation(self, parameters):
        try:
            num_qubits_hamiltonian = self.original_hamiltonian.num_qubits
            transpiled_circuit = self.transpiled_circuit
            
            # Only trim if we have more qubits than needed
            if transpiled_circuit.num_qubits > num_qubits_hamiltonian:
                # Create new quantum and classical registers
                qreg = QuantumRegister(num_qubits_hamiltonian, 'q')
                creg = ClassicalRegister(num_qubits_hamiltonian, 'c')
                trimmed_circuit = QuantumCircuit(qreg, creg)
                
                # Copy only instructions that use first num_qubits_hamiltonian qubits
                for inst in transpiled_circuit.data:
                    # Check if instruction uses only relevant qubits
                    qubit_indices = [transpiled_circuit.find_bit(q).index for q in inst.qubits]
                    if all(idx < num_qubits_hamiltonian for idx in qubit_indices):
                        # Map qubits to new circuit
                        new_qubits = [qreg[idx] for idx in qubit_indices]
                        if inst.clbits:
                            clbit_indices = [transpiled_circuit.find_bit(c).index for c in inst.clbits]
                            new_clbits = [creg[idx] for idx in clbit_indices]
                            trimmed_circuit.append(inst.operation, new_qubits, new_clbits)
                        else:
                            trimmed_circuit.append(inst.operation, new_qubits)
                
                # Use trimmed circuit for evaluation
                transpiled_circuit = trimmed_circuit
            
            circuit_params = transpiled_circuit.parameters
            parameter_dict = dict(zip(circuit_params, parameters))
            pub = (transpiled_circuit, self.original_hamiltonian, parameter_dict)
            
            job = self.estimator.run(pubs=[pub])
            result = job.result()

            expectation_value = float(result[0].data.evs)
            
            self.optimization_history['energies'].append(expectation_value)
            self.optimization_history['parameters'].append(parameters.tolist())
            self.optimization_history['iterations'] += 1
            
            # Enhanced energy status reporting
            current_iter = self.optimization_history['iterations']
            if current_iter == 1:
                print("\n=== VQE Optimization Status ===")
                print(f"Initial Energy: {expectation_value:.8f} Hartree")
            else:
                prev_energy = self.optimization_history['energies'][-2]
                energy_diff = expectation_value - prev_energy
                print(f"\nIteration {current_iter}:")
                print(f"Current Energy: {expectation_value:.8f} Hartree")
                print(f"Energy Change: {energy_diff:+.8f} Hartree")
                print(f"Best Energy So Far: {min(self.optimization_history['energies']):.8f} Hartree")
            
            return expectation_value
        
        except Exception as e:
            logger.error(f"Error in expectation evaluation: {str(e)}")
            raise
    
    def optimize(self, initial_params=None, max_iter=200, optimizer_type='COBYLA', **optimizer_kwargs):
        if initial_params is None:
            initial_params = 2 * np.pi * np.random.random(self.ansatz.num_parameters)
        
        if optimizer_type.upper() == 'COBYLA':
            # Calculate rhobeg based on current cycle (if information is available)
            current_cycle = optimizer_kwargs.get('current_cycle', 0)
            total_cycles = optimizer_kwargs.get('n_cycles', 12)  # Default to 12 if not specified
            
            # Calculate decreasing rhobeg from 1.0 to 0.1 based on cycle progression
            if total_cycles > 1:
                rhobeg = 1.0 - 0.9 * (current_cycle / (total_cycles - 1))
            else:
                rhobeg = 1.0
            rhobeg = optimizer_kwargs.get('rhobeg', rhobeg)
            # Ensure rhobeg doesn't go below 0.1 for numerical stability
            rhobeg = max(0.1, rhobeg)
            
            # Get options from kwargs or use calculated values
            options = {
                'maxiter': max_iter,
                'rhobeg': rhobeg  # Use provided rhobeg or calculated value
            }
            
            print(f"rhobeg: {rhobeg}")

            # Original COBYLA optimization without specified tolerance
            result = minimize(
                self.evaluate_expectation,
                initial_params,
                method='COBYLA',
                options=options
            )
            return result
        else:
            raise ValueError(f"Unsupported optimization method: {optimizer_type}. Use 'COBYLA' or 'ADAM'.")

####################################
########### Hamiltonian ############
####################################

# def to_qiskit_hamiltonian(hamiltonian, noq=None):
#     OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}

#     if noq is None:
#         noq = max([wire for term in hamiltonian.ops for wire in term.wires]) + 1

#     pauli_strings = []
#     coefficients = []

#     for i, obs in enumerate(hamiltonian.ops):
#         pauli_term = ["I"] * noq  # Initialize as identity

#         if isinstance(obs, qml.ops.Prod):  # Tensor product
#             for sub_obs in obs.operands:
#                 pauli_term[sub_obs.wires[0]] = OBS_MAP[sub_obs.name]
#         elif obs.name in OBS_MAP:  # Single Pauli operator
#             pauli_term[obs.wires[0]] = OBS_MAP[obs.name]
#         else:
#             raise ValueError(f"Unsupported observable: {obs} of type {type(obs)}")

#         pauli_strings.append("".join(pauli_term))
#         coefficients.append(hamiltonian.coeffs[i])

#     # Convert to Qiskit SparsePauliOp
#     qiskit_hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))

#     return qiskit_hamiltonian

def to_qiskit_hamiltonian(hamiltonian, noq=None):
    OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}
    
    # Determine number of qubits if not provided
    if noq is None:
        # Get all wires from operators that have them
        all_wires = []
        for term in hamiltonian.ops:
            if hasattr(term, 'wires') and len(term.wires) > 0:
                all_wires.extend(list(term.wires))
        
        if all_wires:
            noq = max(all_wires) + 1
        else:
            # Default to 1 qubit if no wires found
            noq = 1
    
    pauli_strings = []
    coefficients = []
    
    for i, obs in enumerate(hamiltonian.ops):
        # Handle global Identity operator (with no wires)
        if obs.name == "Identity" and (not hasattr(obs, 'wires') or len(obs.wires) == 0):
            # For global identity, add an all-identity string
            pauli_strings.append("I" * noq)
            coefficients.append(hamiltonian.coeffs[i])
            continue
        
        pauli_term = ["I"] * noq  # Initialize as identity
        
        if isinstance(obs, qml.ops.Prod):  # Tensor product
            for sub_obs in obs.operands:
                if hasattr(sub_obs, 'wires') and len(sub_obs.wires) > 0:
                    pauli_term[sub_obs.wires[0]] = OBS_MAP[sub_obs.name]
        
        elif obs.name in OBS_MAP:  # Single Pauli operator
            if hasattr(obs, 'wires') and len(obs.wires) > 0:
                pauli_term[obs.wires[0]] = OBS_MAP[obs.name]
            else:
                raise ValueError(f"Non-identity observable {obs} (index {i}) has no wires")
        
        else:
            raise ValueError(f"Unsupported observable: {obs} of type {type(obs)}")
        
        pauli_strings.append("".join(pauli_term))
        coefficients.append(hamiltonian.coeffs[i])
    
    # Convert to Qiskit SparsePauliOp
    qiskit_hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))
    return qiskit_hamiltonian

####################################
############## Utils ###############
####################################

def load_backend_data(mappings_file: str) -> List[Dict]:
    """Load pre-generated backend data from pickle file."""
    if not mappings_file.endswith('.pkl'):
        mappings_file += '.pkl'
    
    try:
        with open(mappings_file, 'rb') as f:
            data = pickle.load(f)
        mappings_data = data['mappings_data']
        metadata = data['metadata']
        logger.info(f"Loaded {len(mappings_data)} mappings from {mappings_file}")
        return data, mappings_data, metadata
    except Exception as e:
        logger.error(f"Error loading mappings: {str(e)}")
        raise
    
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
    
####################################
########### Save Results ############
####################################

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
    with plot_lock:
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
    with plot_lock:
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
    with plot_lock:
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
    with plot_lock:
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

def plot_multi_run_comparison(output_dir: str, num_runs: int, date: str, backend_name: str, schedule_type: str):
    """Create comparison plots across multiple runs."""
    # Create a directory for aggregate results
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Collect energy history data from all runs
    all_energies = []
    for run_index in range(num_runs):
        run_dir = os.path.join(output_dir, f"run_{run_index}")
        # Look for summary json files
        summary_files = [f for f in os.listdir(run_dir) if f.endswith("_summary.json")]
        
        if not summary_files:
            print(f"Warning: No summary files found for run {run_index}")
            continue
            
        # Use the first summary file (should only be one per run)
        summary_path = os.path.join(run_dir, summary_files[0])
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            energies = [item["energy"] for item in summary_data["energies_per_iteration"]]
            all_energies.append(energies)
    
    # 1. Plot all runs on the same graph
    with plot_lock:
        plt.figure(figsize=(12, 8))
        max_iterations = max([len(energies) for energies in all_energies])
        iterations = list(range(1, max_iterations + 1))
        
        for run_index, energies in enumerate(all_energies):
            plt.plot(iterations[:len(energies)], energies, alpha=0.7, label=f'Run {run_index}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title(f'Energy Convergence Across All Runs: {backend_name} ({date}) - {schedule_type}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(aggregate_dir, f'all_runs_energy_convergence_{schedule_type}.png'), dpi=300)
        plt.close()
    
    # 2. Plot best energy achieved by each run
    best_energies = [min(energies) for energies in all_energies]
    with plot_lock:
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_runs), best_energies)
        plt.axhline(y=np.mean(best_energies), color='r', linestyle='-', label=f'Mean: {np.mean(best_energies):.6f}')
        plt.axhline(y=np.mean(best_energies) + np.std(best_energies), color='g', linestyle='--', 
                    label=f'Mean + StdDev: {np.mean(best_energies) + np.std(best_energies):.6f}')
        plt.axhline(y=np.mean(best_energies) - np.std(best_energies), color='g', linestyle='--',
                    label=f'Mean - StdDev: {np.mean(best_energies) - np.std(best_energies):.6f}')
        
        plt.xlabel('Run Index')
        plt.ylabel('Best Energy')
        plt.title(f'Best Energy Achieved Per Run: {backend_name} ({date}) - {schedule_type}')
        plt.xticks(range(num_runs))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(aggregate_dir, f'best_energy_per_run_{schedule_type}.png'), dpi=300)
        plt.close()
    
    # 3. Statistical box plot of energy convergence
    with plot_lock:
        plt.figure(figsize=(12, 8))
        
        # Calculate median energy trajectory and quartiles
        min_length = min([len(e) for e in all_energies])
        aligned_energies = [e[:min_length] for e in all_energies]
        energy_array = np.array(aligned_energies)
        
        median_energy = np.median(energy_array, axis=0)
        q1_energy = np.percentile(energy_array, 25, axis=0)
        q3_energy = np.percentile(energy_array, 75, axis=0)
        min_energy = np.min(energy_array, axis=0)
        max_energy = np.max(energy_array, axis=0)
        
        iterations = list(range(1, min_length + 1))
        
        plt.plot(iterations, median_energy, 'b-', linewidth=2, label='Median')
        plt.fill_between(iterations, q1_energy, q3_energy, color='b', alpha=0.2, label='IQR (25-75%)')
        plt.fill_between(iterations, min_energy, max_energy, color='b', alpha=0.1, label='Min-Max Range')
        
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title(f'Statistical Distribution of Energy Convergence: {backend_name} ({date}) - {schedule_type}')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.savefig(os.path.join(aggregate_dir, f'energy_statistics_{schedule_type}.png'), dpi=300)
    plt.close()

def compile_multi_run_stats(output_dir: str, num_runs: int, date: str, backend_name: str, schedule_type: str):
    """Compile statistics across multiple runs and create aggregate plots."""
    # Create a directory for aggregate results
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    
    # Collect data from all runs
    all_run_data = []
    for run_index in range(num_runs):
        run_dir = os.path.join(output_dir, f"run_{run_index}")
        # Look for summary json files
        summary_files = [f for f in os.listdir(run_dir) if f.endswith("_summary.json")]
        
        if not summary_files:
            print(f"Warning: No summary files found for run {run_index}")
            continue
            
        # Use the first summary file (should only be one per run)
        summary_path = os.path.join(run_dir, summary_files[0])
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            all_run_data.append(summary_data)
    
    # Extract key performance metrics
    best_energies = [data.get('best_energy') for data in all_run_data]
    final_energies = [data.get('final_energy') for data in all_run_data]
    run_times = [data.get('run_time') for data in all_run_data]
    iterations_to_best = [data.get('best_energy_iteration') for data in all_run_data]
    
    # Calculate statistics and convert to native Python types
    stats = {
        "best_energy": {
            "mean": float(np.mean(best_energies)),
            "std": float(np.std(best_energies)),
            "min": float(np.min(best_energies)),
            "max": float(np.max(best_energies))
        },
        "final_energy": {
            "mean": float(np.mean(final_energies)),
            "std": float(np.std(final_energies)),
            "min": float(np.min(final_energies)),
            "max": float(np.max(final_energies))
        },
        "run_time": {
            "mean": float(np.mean(run_times)),
            "std": float(np.std(run_times)),
            "min": float(np.min(run_times)),
            "max": float(np.max(run_times))
        },
        "iterations_to_best": {
            "mean": float(np.mean(iterations_to_best)),
            "std": float(np.std(iterations_to_best)),
            "min": int(np.min(iterations_to_best)),
            "max": int(np.max(iterations_to_best))
        }
    }
    
    # Save aggregate statistics as JSON
    stats_path = os.path.join(aggregate_dir, f"{backend_name}_{date}_{schedule_type}_aggregate_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create a readable text summary
    txt_path = os.path.join(aggregate_dir, f"{backend_name}_{date}_{schedule_type}_aggregate_stats.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Aggregate Statistics for {num_runs} Runs\n")
        f.write(f"=====================================\n\n")
        f.write(f"Date: {date}\n")
        f.write(f"Backend: {backend_name}\n")
        f.write(f"Schedule type: {schedule_type}\n\n")
        
        f.write(f"Best Energy Statistics\n")
        f.write(f"---------------------\n")
        f.write(f"Mean: {stats['best_energy']['mean']:.8f}\n")
        f.write(f"Standard Deviation: {stats['best_energy']['std']:.8f}\n")
        f.write(f"Min: {stats['best_energy']['min']:.8f}\n")
        f.write(f"Max: {stats['best_energy']['max']:.8f}\n\n")
        
        f.write(f"Final Energy Statistics\n")
        f.write(f"----------------------\n")
        f.write(f"Mean: {stats['final_energy']['mean']:.8f}\n")
        f.write(f"Standard Deviation: {stats['final_energy']['std']:.8f}\n")
        f.write(f"Min: {stats['final_energy']['min']:.8f}\n")
        f.write(f"Max: {stats['final_energy']['max']:.8f}\n\n")
        
        f.write(f"Run Time Statistics (seconds)\n")
        f.write(f"----------------------------\n")
        f.write(f"Mean: {stats['run_time']['mean']:.2f}\n")
        f.write(f"Standard Deviation: {stats['run_time']['std']:.2f}\n")
        f.write(f"Min: {stats['run_time']['min']:.2f}\n")
        f.write(f"Max: {stats['run_time']['max']:.2f}\n\n")
        
        f.write(f"Iterations to Best Energy Statistics\n")
        f.write(f"--------------------------------\n")
        f.write(f"Mean: {stats['iterations_to_best']['mean']:.1f}\n")
        f.write(f"Standard Deviation: {stats['iterations_to_best']['std']:.1f}\n")
        f.write(f"Min: {stats['iterations_to_best']['min']:.0f}\n")
        f.write(f"Max: {stats['iterations_to_best']['max']:.0f}\n\n")
        
        # Write per-run data
        f.write(f"Individual Run Data\n")
        f.write(f"-----------------\n")
        f.write(f"{'Run':^5}{'Best Energy':^20}{'Final Energy':^20}{'Run Time (s)':^15}{'Iter to Best':^15}\n")
        for i, data in enumerate(all_run_data):
            f.write(f"{i:^5}{data['best_energy']:^20.8f}{data['final_energy']:^20.8f}{data['run_time']:^15.2f}{data['best_energy_iteration']:^15.0f}\n")
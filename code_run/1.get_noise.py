import os
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, List
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_directory() -> str:
    today = datetime.now()
    output_dir = f"noise_models/{today.strftime('%m_%d')}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def get_noise_model(service: QiskitRuntimeService, backend_name: str) -> Tuple[NoiseModel, Dict, List]:
    try:
        logger.info(f"Getting noise model from {backend_name}")
        
        # Get backend
        backend = service.backend(backend_name)
        
        # Get noise model
        noise_model = NoiseModel.from_backend(backend)
        
        # Get configuration and properties
        config = backend.configuration()
        properties = backend.properties()
        
        # Extract coupling map and basis gates
        coupling_map = config.coupling_map
        basis_gates = config.basis_gates
        
        # Extract T1 and T2 times
        t1_times = {}
        t2_times = {}
        
        # Extract measurement error rates
        meas_error_rates = {}
        
        # Extract relaxation times and measurement errors from properties
        for qubit_idx in range(config.n_qubits):
            try:
                t1_times[qubit_idx] = properties.t1(qubit_idx)
                t2_times[qubit_idx] = properties.t2(qubit_idx)
                meas_error_rates[qubit_idx] = properties.readout_error(qubit_idx)
            except Exception as e:
                logger.warning(f"Could not extract T1/T2/measurement error for qubit {qubit_idx}: {str(e)}")
                t1_times[qubit_idx] = 0
                t2_times[qubit_idx] = 0
                meas_error_rates[qubit_idx] = 0
        
        # Create properties dictionary
        properties_dict = {
            "coupling_map": coupling_map,
            "basis_gates": basis_gates,
            "relaxation_times": {
                "t1_times": t1_times,
                "t2_times": t2_times
            },
            "measurement_errors": meas_error_rates,
            "backend_name": backend_name,
            "date_retrieved": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully retrieved noise model and properties for {backend_name}")
        return noise_model, properties_dict, basis_gates
        
    except Exception as e:
        logger.error(f"Error getting noise model for {backend_name}: {str(e)}")
        raise

def save_noise_data(output_dir: str, backend_name: str, 
                   noise_model: NoiseModel, properties: Dict) -> None:
    try:
        # Create backend-specific directory
        backend_dir = os.path.join(output_dir, backend_name)
        os.makedirs(backend_dir, exist_ok=True)
        
        # Save noise model
        noise_model_path = os.path.join(backend_dir, 'noise_model.pkl')
        with open(noise_model_path, 'wb') as f:
            pickle.dump(noise_model, f)
        
        # Save properties
        properties_path = os.path.join(backend_dir, 'properties.json')
        with open(properties_path, 'w') as f:
            json.dump(properties, f, indent=2)
        
        logger.info(f"Saved noise data for {backend_name} in {backend_dir}")
    except Exception as e:
        logger.error(f"Error saving noise data for {backend_name}: {str(e)}")
        raise

def main():
    try:
        backend_names = [
            "ibm_brisbane", 
            "ibm_brussels", 
            "ibm_sherbrooke", 
            "ibm_strasbourg"
        ]
        
        # Create output directory with today's date
        output_dir = create_output_directory()
        
        service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")        
        # Get and save noise models for each backend
        for backend_name in backend_names:
            try:
                noise_model, properties, basis_gates = get_noise_model(service, backend_name)
                save_noise_data(output_dir, backend_name, noise_model, properties)
                logger.info(f"Successfully processed {backend_name}")
            except Exception as e:
                logger.error(f"Failed to process {backend_name}: {str(e)}")
                continue
        
        logger.info(f"Completed noise model collection for all backends")
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 

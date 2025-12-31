# Three Birds with One Stone: Improving Performance, Convergence, and System Throughput with NEST.

This repository contains the artifacts and experimental data for the paper "Three Birds with One Stone: Improving Performance, Convergence, and System Throughput with NEST"

## Structure

```
├── code_run/           # Main Nest execution scripts
├── data/              # Experimental data
```

## Environment

### Requirements
- Python 3.12+
- Qiskit 1.4.0
- PennyLane
- NumPy, SciPy, Matplotlib

## Usage

```python
# 1. Get noise models
python code_run/1.get_noise.py

# 2. Generate mappings
python code_run/2.generate_maps.py

# 3. Run experiments
python code_run/3.run_nest_single.py
python code_run/4.run_nest_parallel.py
```

### Batch Scripts
```bash
./code_run/3.run_nest_single.sh     # Single experiments
./code_run/4.run_nest_parallel.sh   # Parallel experiments  
./code_run/7.run_all_schedules.sh   # All schedules
```

# Code Licensing
© 2025 Rice University subject to Creative Commons Attribution 4.0 International license (Creative Commons — Attribution 4.0 International — CC BY 4.0)

Contact ptl@rice.edu for permissions.
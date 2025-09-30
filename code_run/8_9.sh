#!/bin/bash

echo "Starting ablation cycle script..."
bash 8.albation_cycle.sh

echo "==============================================="
echo "Cycle ablation completed. Starting iteration ablation..."
echo "==============================================="

bash 9.ablation_iteration.sh

echo "All ablation tests completed successfully!"

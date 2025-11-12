#!/bin/bash

# Test FreqMixFormer on MMASD Dataset
# This script tests a trained model on the validation set

# Configuration
CONFIG="./config/mmasd/train_joint.yaml"
WORK_DIR="./work_dir/mmasd_joint"

# Check if weights file is provided
if [ -z "$1" ]; then
    echo "Usage: ./test_mmasd.sh <path_to_weights.pt>"
    echo "Example: ./test_mmasd.sh ./work_dir/mmasd_joint/runs-50-12345.pt"
    exit 1
fi

WEIGHTS=$1

# Check if weights file exists
if [ ! -f "$WEIGHTS" ]; then
    echo "Error: Weights file not found: $WEIGHTS"
    exit 1
fi

# Test the model
python main.py \
    --config $CONFIG \
    --phase test \
    --weights $WEIGHTS \
    --device 0 \
    --test-batch-size 32 \
    --save-score True \
    --work-dir $WORK_DIR

echo "Testing completed! Check results in $WORK_DIR"


#!/bin/bash

# Training script for MMASD dataset with FreqMixFormer

# Set CUDA device (change as needed)
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "FreqMixFormer Training on MMASD Dataset"
echo "============================================================"

# Parse command line arguments
MODE=${1:-"joint"}  # Default to joint modality
PHASE=${2:-"train"}  # Default to train phase
WEIGHTS=${3:-""}

case $MODE in
    "joint")
        CONFIG="./config/mmasd/train_joint.yaml"
        echo "Training with JOINT modality"
        ;;
    "bone")
        CONFIG="./config/mmasd/train_bone.yaml"
        echo "Training with BONE modality"
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Usage: $0 [joint|bone] [train|test] [weights_path]"
        echo ""
        echo "Examples:"
        echo "  $0 joint train                    # Train joint model"
        echo "  $0 bone train                     # Train bone model"
        echo "  $0 joint test weights.pt          # Test joint model"
        exit 1
        ;;
esac

echo "Config: $CONFIG"
echo "Phase: $PHASE"

if [ "$PHASE" == "test" ]; then
    if [ -z "$WEIGHTS" ]; then
        echo "Error: Weights path required for testing"
        echo "Usage: $0 $MODE test <weights_path>"
        exit 1
    fi
    echo "Weights: $WEIGHTS"
    echo ""
    python main.py --config $CONFIG --phase test --weights $WEIGHTS
else
    echo ""
    python main.py --config $CONFIG --phase train
fi

echo ""
echo "============================================================"
echo "Training/Testing completed!"
echo "============================================================"


#!/bin/bash
# Script to visualize the same sample with both visualization scripts

# Check if a sample file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_csv_file>"
    echo ""
    echo "Example:"
    echo "  $0 '3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/Arm_Swing/processed_Arm_Swing_P1_R1_0.csv'"
    echo ""
    echo "Or use without argument to visualize the first available sample:"
    SAMPLE=$(find "3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit" -name "*.csv" -type f | head -n 1)
else
    SAMPLE="$1"
fi

# Check if sample exists
if [ ! -f "$SAMPLE" ]; then
    echo "Error: File not found: $SAMPLE"
    exit 1
fi

echo "======================================================================"
echo "Visualizing the same sample with both scripts"
echo "Sample: $SAMPLE"
echo "======================================================================"
echo ""

echo "1. Running visualize_demo.py..."
echo "----------------------------------------------------------------------"
python visualize_demo.py "$SAMPLE"

echo ""
echo "2. Running visualize_enhanced_mmasd_3d.py..."
echo "----------------------------------------------------------------------"
python visualize_enhanced_mmasd_3d.py "$SAMPLE"

echo ""
echo "======================================================================"
echo "Both visualizations complete!"
echo "Check the ./visualizations/ directory for output images."
echo "======================================================================"


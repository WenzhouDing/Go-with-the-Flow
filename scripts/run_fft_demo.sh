#!/bin/bash
# Demo: Extract motion and apply FFT-based frequency filtering

set -e

echo "================================================================================"
echo "DEMO: Motion Extraction + FFT-Based Frequency Filtering"
echo "================================================================================"
echo ""

# Source video
SOURCE_VIDEO="my_warped_noise.mp4"
OUTPUT_DIR="fft_demo_output_$(date +%Y%m%d_%H%M%S)"

echo "Step 1: Extracting optical flow from video..."
echo "Source: $SOURCE_VIDEO"
echo "Output: $OUTPUT_DIR"
echo ""

# Extract flow using our implementation
conda activate flow_warp
python extract_flow_sequence.py "$SOURCE_VIDEO" \
    --output "${OUTPUT_DIR}/flows.npy" \
    --fps 12

echo ""
echo "✓ Flow extraction complete!"
echo ""

echo "Step 2: Applying FFT-based frequency decomposition..."
echo "  - Decomposing into 4 frequency bands"
echo "  - Analyzing motion patterns"
echo "  - Identifying camera shake vs. body movement"
echo ""

# Apply frequency filtering
conda activate flow_warp
python frequency_motion_transfer.py "$SOURCE_VIDEO" \
    --flow_path "${OUTPUT_DIR}/flows.npy" \
    --output_dir "$OUTPUT_DIR" \
    --num_bands 4 \
    --remove_shake \
    --fps 12

echo ""
echo "✓ Frequency filtering complete!"
echo ""

echo "Step 3: Results:"
echo "  - Raw flows: $OUTPUT_DIR/raw_flows.npy"
echo "  - Clean flow: $OUTPUT_DIR/clean_flow.npy"
echo "  - Analysis: $OUTPUT_DIR/motion_analysis.json"
echo "  - Individual bands: $OUTPUT_DIR/band_*_flow.npy"
echo ""

echo "Step 4: Visualizing frequency decomposition..."
conda activate flow_warp
python test_frequency_decomposition.py "$OUTPUT_DIR" \
    --output "${OUTPUT_DIR}/visualizations"

echo ""
echo "================================================================================"
echo "DEMO COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"/ | grep -E "^-" | head -10
echo ""
echo "Visualizations:"
ls -lh "$OUTPUT_DIR/visualizations/"*.png 2>/dev/null || echo "  (visualizations will be in ${OUTPUT_DIR}/visualizations/)"
echo ""


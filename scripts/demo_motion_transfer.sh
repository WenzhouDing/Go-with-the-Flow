#!/bin/bash
# demo_motion_transfer.sh
# Complete demonstration of Go-with-the-Flow motion transfer pipeline

set -e  # Exit on error

echo "================================================================================"
echo "Go-with-the-Flow Motion Transfer Pipeline - Live Demo"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# PART 1: Using Existing Warped Noise Data
# ============================================================================

echo -e "${BLUE}PART 1: Generate Video Using Existing Motion Data${NC}"
echo "--------------------------------------------------------------------------------"
echo ""

if [ -d "my_warped_noise" ] && [ -f "my_warped_noise/noises.npy" ]; then
    echo -e "${GREEN}✓ Found existing warped noise data in my_warped_noise/${NC}"
    echo ""
    echo "Files in my_warped_noise/:"
    ls -lh my_warped_noise/ | grep -E "^-|^d"
    echo ""

    echo -e "${YELLOW}To generate a video with this motion:${NC}"
    echo ""
    echo "conda activate flow"
    echo "python cut_and_drag_inference.py my_warped_noise \\"
    echo "    --prompt \"A cartoon robot dancing with LED lights\" \\"
    echo "    --output_mp4_path robot_dance.mp4 \\"
    echo "    --device cuda \\"
    echo "    --num_inference_steps 30"
    echo ""
    echo "Press Enter to continue to Part 2, or Ctrl+C to exit and run the above command..."
    read
else
    echo "No existing warped noise found. Will demonstrate full pipeline in Part 2."
    echo ""
fi

# ============================================================================
# PART 2: Extract Motion from a New Video
# ============================================================================

echo ""
echo -e "${BLUE}PART 2: Extract Motion from a Video${NC}"
echo "--------------------------------------------------------------------------------"
echo ""

# Define test video URL (short, public domain)
# This is a short clip from Pixabay (public domain)
TEST_VIDEO_URL="https://cdn.pixabay.com/video/2023/08/05/174963-851965097_tiny.mp4"

echo "We'll use a short public domain video from Pixabay:"
echo "URL: $TEST_VIDEO_URL"
echo ""

OUTPUT_FOLDER="demo_motion_$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}Command to extract motion:${NC}"
echo ""
echo "conda activate flow_warp"
echo "python make_warped_noise.py \"$TEST_VIDEO_URL\" \\"
echo "    --output_folder $OUTPUT_FOLDER"
echo ""

echo "This will:"
echo "  1. Download and preprocess the video (resize to 480x720, 49 frames)"
echo "  2. Extract optical flow using RAFT"
echo "  3. Create temporally consistent warped noise"
echo "  4. Save results to $OUTPUT_FOLDER/"
echo ""

echo -e "${YELLOW}Expected outputs:${NC}"
echo "  - noises.npy         : Warped noise (main output)"
echo "  - flows_dxdy.npy     : Optical flow fields"
echo "  - input.mp4          : Preprocessed input video"
echo "  - first_frame.png    : First frame reference"
echo "  - noise_video.mp4    : Visualization"
echo ""

echo "Press Enter to see Part 3, or Ctrl+C to exit..."
read

# ============================================================================
# PART 3: Generate New Video with Transferred Motion
# ============================================================================

echo ""
echo -e "${BLUE}PART 3: Generate Video with Transferred Motion${NC}"
echo "--------------------------------------------------------------------------------"
echo ""

echo -e "${YELLOW}After extracting motion, generate a new video:${NC}"
echo ""
echo "conda activate flow"
echo "python cut_and_drag_inference.py $OUTPUT_FOLDER \\"
echo "    --prompt \"A watercolor painting of dancers in motion\" \\"
echo "    --output_mp4_path watercolor_dance.mp4 \\"
echo "    --device cuda \\"
echo "    --num_inference_steps 30"
echo ""

echo "This uses CogVideoX to generate a video where the motion pattern is"
echo "transferred but the visual content matches your prompt."
echo ""

# ============================================================================
# PART 4: Alternative Options
# ============================================================================

echo ""
echo -e "${BLUE}PART 4: Alternative Options${NC}"
echo "--------------------------------------------------------------------------------"
echo ""

echo -e "${YELLOW}Option A: Image-to-Video (I2V) with motion transfer${NC}"
echo ""
echo "python cut_and_drag_inference.py $OUTPUT_FOLDER \\"
echo "    --image_path your_image.png \\"
echo "    --prompt \"Animate this image with dancing motion\" \\"
echo "    --output_mp4_path animated.mp4 \\"
echo "    --device cuda"
echo ""

echo -e "${YELLOW}Option B: Control motion strength with degradation${NC}"
echo ""
echo "# Full motion transfer (0.0)"
echo "python cut_and_drag_inference.py $OUTPUT_FOLDER \\"
echo "    --prompt \"A dancing flame\" \\"
echo "    --degradation 0.0 \\"
echo "    --output_mp4_path full_motion.mp4"
echo ""
echo "# Partial motion transfer (0.5)"
echo "python cut_and_drag_inference.py $OUTPUT_FOLDER \\"
echo "    --prompt \"A dancing flame\" \\"
echo "    --degradation 0.5 \\"
echo "    --output_mp4_path partial_motion.mp4"
echo ""

echo -e "${YELLOW}Option C: Use local video file${NC}"
echo ""
echo "python make_warped_noise.py /path/to/your/video.mp4 \\"
echo "    --output_folder my_custom_motion"
echo ""

# ============================================================================
# PART 5: Complete Example Commands
# ============================================================================

echo ""
echo -e "${BLUE}PART 5: Ready-to-Run Example Commands${NC}"
echo "--------------------------------------------------------------------------------"
echo ""

cat << 'EOF'
# Example 1: Using existing warped noise
conda activate flow
python cut_and_drag_inference.py my_warped_noise \
    --prompt "A futuristic hologram dancing" \
    --output_mp4_path hologram.mp4 \
    --device cuda \
    --num_inference_steps 30

# Example 2: Full pipeline with public domain video
conda activate flow_warp
python make_warped_noise.py \
    "https://cdn.pixabay.com/video/2023/08/05/174963-851965097_tiny.mp4" \
    --output_folder dancing_motion

conda activate flow
python cut_and_drag_inference.py dancing_motion \
    --prompt "A cartoon character dancing energetically" \
    --output_mp4_path cartoon_dance.mp4 \
    --device cuda

# Example 3: Using your own video file
conda activate flow_warp
python make_warped_noise.py my_video.mp4 \
    --output_folder my_motion

conda activate flow
python cut_and_drag_inference.py my_motion \
    --image_path starting_image.png \
    --prompt "Animate this with the video's motion" \
    --output_mp4_path my_result.mp4 \
    --device cuda

# Example 4: Quick test with existing data
conda activate flow
python cut_and_drag_inference.py my_warped_noise \
    --prompt "A dancing skeleton made of flames" \
    --output_mp4_path test_output.mp4 \
    --device cuda \
    --num_inference_steps 20
EOF

echo ""
echo "================================================================================"
echo "Demo Complete!"
echo "================================================================================"
echo ""
echo "Quick Start:"
echo "  1. Use existing motion data: See Example 1 above"
echo "  2. Extract from new video: See Examples 2-3 above"
echo "  3. Read full guide: MOTION_TRANSFER_DEMO.md"
echo ""
echo "Two environments needed:"
echo "  - flow_warp: For make_warped_noise.py (motion extraction)"
echo "  - flow: For cut_and_drag_inference.py (video generation)"
echo ""

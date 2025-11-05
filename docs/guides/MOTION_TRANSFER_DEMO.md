# Motion Transfer Demo Guide

## Basic Go-with-the-Flow Motion Transfer Pipeline

This guide demonstrates how to transfer motion from one video to generate a new video.

## Prerequisites

**Two Conda Environments:**
1. `flow_warp` - For optical flow extraction and noise warping
2. `flow` - For video diffusion generation

## Pipeline Overview

```
Source Video → Extract Motion → Warp Noise → Generate Video
    ↓              (RAFT)         (flow-based)    (Diffusion)
[make_warped_noise.py]         [cut_and_drag_inference.py]
```

## Step-by-Step Workflow

### Method 1: Using an Existing Video File

#### Step 1: Extract Motion and Create Warped Noise
```bash
# Activate the flow_warp environment
conda activate flow_warp

# Extract motion from source video
python make_warped_noise.py <VIDEO_PATH> --output_folder demo_motion_output

# Example with local file:
python make_warped_noise.py my_warped_noise.mp4 --output_folder demo_motion_output
```

**What this does:**
- Loads and preprocesses video to 480x720, 49 frames
- Extracts optical flow using RAFT
- Creates temporally consistent warped noise
- Saves to `demo_motion_output/noises.npy`
- Creates preview visualizations

**Expected Output:**
```
demo_motion_output/
├── noises.npy              # The warped noise (main output)
├── flows.npy               # Optical flow fields
├── preview.mp4             # Visualization of warped noise
├── first_frame.png         # First frame reference
└── input.mp4               # Preprocessed input video
```

#### Step 2: Generate New Video with Transferred Motion
```bash
# Switch to flow environment
conda activate flow

# Generate video using text prompt
python cut_and_drag_inference.py demo_motion_output \
    --prompt "A cartoon duck dancing" \
    --output_mp4_path duck_dancing.mp4 \
    --device cuda \
    --num_inference_steps 30

# Or use an image for I2V (Image-to-Video)
python cut_and_drag_inference.py demo_motion_output \
    --image_path my_image.png \
    --prompt "A robot performing choreography" \
    --output_mp4_path robot_dance.mp4 \
    --device cuda \
    --num_inference_steps 30
```

**What this does:**
- Loads the warped noise from demo_motion_output
- Uses CogVideoX diffusion model
- Generates 49 frames with the motion pattern
- Outputs final video

### Method 2: Using a YouTube Video URL

```bash
conda activate flow_warp

# Direct URL input (no download needed)
python make_warped_noise.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output_folder youtube_motion
```

The pipeline automatically handles URL downloads.

### Method 3: Using Example from Repo

```bash
# Check what example videos/outputs exist
ls -lh my_warped_noise/
ls -lh outputs/

# If my_warped_noise/ folder exists with noises.npy:
conda activate flow
python cut_and_drag_inference.py my_warped_noise \
    --prompt "A watercolor painting of a dancing figure" \
    --output_mp4_path watercolor_dance.mp4 \
    --device cuda
```

## Quick Demo Script

Here's a complete demo script you can run:

```bash
#!/bin/bash

echo "=== Go-with-the-Flow Motion Transfer Demo ==="

# Step 1: Activate flow_warp environment
echo "Step 1: Extracting motion from source video..."
conda activate flow_warp

# Use existing video or provide your own
SOURCE_VIDEO="my_warped_noise.mp4"
OUTPUT_FOLDER="demo_transfer_$(date +%s)"

python make_warped_noise.py "$SOURCE_VIDEO" \
    --output_folder "$OUTPUT_FOLDER"

echo "✓ Motion extracted to $OUTPUT_FOLDER"

# Step 2: Generate new video
echo "Step 2: Generating video with transferred motion..."
conda activate flow

python cut_and_drag_inference.py "$OUTPUT_FOLDER" \
    --prompt "A futuristic robot dancing with LED lights" \
    --output_mp4_path "robot_dance_output.mp4" \
    --device cuda \
    --num_inference_steps 30

echo "✓ Generated video: robot_dance_output.mp4"
echo "=== Demo Complete! ==="
```

## Parameters Explained

### make_warped_noise.py

**Basic Usage:**
```bash
python make_warped_noise.py <VIDEO_PATH> --output_folder <OUTPUT_DIR>
```

**Advanced Options:**
- Video can be: local file path, URL, or numpy array
- Automatically resizes to 480x720, 49 frames
- Uses RAFT for optical flow extraction
- Creates latent-space warped noise (16 channels)

### cut_and_drag_inference.py

**Key Parameters:**
```bash
python cut_and_drag_inference.py <NOISE_FOLDER> \
    --prompt "Text description" \           # T2V: Text prompt
    --image_path "image.png" \              # I2V: Starting image
    --output_mp4_path "output.mp4" \        # Output location
    --device cuda \                          # GPU device
    --num_inference_steps 30 \              # Quality (higher = better)
    --degradation 0.0                       # Motion strength (0=full, 1=none)
```

**Model Selection:**
- Default uses I2V5B (Image-to-Video, 5B params)
- Can specify different LoRA models
- Supports both T2V (text-to-video) and I2V modes

## Degradation Control

Control how much motion is transferred:

```bash
# Full motion transfer (recommended)
--degradation 0.0

# Balanced (50% motion, 50% random)
--degradation 0.5

# Mostly random (minimal motion transfer)
--degradation 0.9
```

## Common Test Videos

Here are some good source videos for motion transfer:

**Dance Videos:**
- Dancing people (transfers choreography)
- Ballet, hip-hop, contemporary dance

**Nature:**
- Waves, water flowing
- Trees swaying in wind
- Clouds moving

**Camera Motion:**
- Camera pans, zooms
- Drone footage
- Rotating objects

**Sports:**
- Running, jumping
- Ball movements
- Athletic activities

## Troubleshooting

### Issue: "output_folder already exists"
```bash
# Solution: Use a different output folder or delete the old one
rm -rf demo_motion_output
python make_warped_noise.py video.mp4 --output_folder demo_motion_output
```

### Issue: CUDA out of memory
```bash
# Solution 1: Use lower resolution (edit make_warped_noise.py LATENT=8)
# Solution 2: Reduce inference steps
python cut_and_drag_inference.py folder --num_inference_steps 20
```

### Issue: "No module named 'rp'"
```bash
# Solution: Install in flow_warp environment
conda activate flow_warp
pip install rp
```

## Example Results

**Input:** Dance video of person doing choreography
**Output Options:**
1. "A robot dancing" → Robot performing same moves
2. "A cartoon character dancing" → Animated version
3. "A watercolor painting in motion" → Artistic style
4. "Flames dancing" → Abstract motion transfer

## Next Steps

After basic motion transfer:
1. Experiment with different prompts
2. Try image-to-video with custom images
3. Adjust degradation for motion strength
4. Use frequency decomposition for advanced control (see QUICK_START_FREQUENCY.md)

## Performance Notes

**make_warped_noise.py:**
- Time: ~1-3 minutes for 49 frames
- Memory: ~2-4GB GPU for RAFT
- Environment: flow_warp

**cut_and_drag_inference.py:**
- Time: ~5-15 minutes depending on model/steps
- Memory: ~8-16GB GPU for CogVideoX-5B
- Environment: flow

## Full Example Session

```bash
# 1. Check environment
conda activate flow_warp
python --version  # Should have rp package

# 2. Use existing video in repo
ls my_warped_noise.mp4  # Check it exists

# 3. Extract motion
python make_warped_noise.py my_warped_noise.mp4 \
    --output_folder test_motion

# Expected output:
# ✓ Noise shape: (49, 16, 60, 90)
# ✓ Flow shape: (48, 2, 480, 720)
# ✓ Output folder: test_motion

# 4. Generate new video
conda activate flow
python cut_and_drag_inference.py test_motion \
    --prompt "A dancing skeleton made of fire" \
    --output_mp4_path fire_skeleton.mp4 \
    --device cuda \
    --num_inference_steps 30

# 5. View result
# Output will be: fire_skeleton.mp4
```

## Success!

You've successfully transferred motion from one video to create a new video!

The motion patterns, timing, and dynamics are preserved while the visual content
is completely transformed according to your prompt.

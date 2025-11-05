# Testing Motion Transfer Pipeline

## Your Existing Data

You already have extracted motion in `my_warped_noise/`:
- **Input video:** 720x480, 4 seconds, 12fps
- **Warped noise:** Ready to use
- **First frame:** Available as reference

## Quick Test (2 minutes)

Test the pipeline with your existing motion data:

```bash
# Activate the flow environment
conda activate flow

# Generate a test video
python cut_and_drag_inference.py my_warped_noise \
    --prompt "A dancing flame with orange and yellow colors" \
    --output_mp4_path flame_test.mp4 \
    --device cuda \
    --num_inference_steps 20
```

**Expected time:** ~2-5 minutes
**Output:** `flame_test.mp4` - 49 frames with the motion from your input video

## Extract Motion from a New Video

### Option A: Use a Direct URL

```bash
# Activate flow_warp environment
conda activate flow_warp

# Public domain video from Pixabay (short dance clip)
python make_warped_noise.py \
    "https://cdn.pixabay.com/video/2023/08/05/174963-851965097_tiny.mp4" \
    --output_folder dance_motion_test

# Expected output:
# ✓ Preprocessed to 480x720, 49 frames
# ✓ Extracted optical flow
# ✓ Created warped noise
# ✓ Saved to dance_motion_test/
```

**Time:** ~1-3 minutes
**Output folder:** `dance_motion_test/`

### Option B: Use Local Video

```bash
conda activate flow_warp

# Use any local video file
python make_warped_noise.py /path/to/your/video.mp4 \
    --output_folder my_motion_test
```

### Option C: Use Another Public Video

Here are some good test URLs (all public domain/Creative Commons):

```bash
# Dancing (Pixabay)
URL1="https://cdn.pixabay.com/video/2023/08/05/174963-851965097_tiny.mp4"

# Nature/Water (Pixabay)
URL2="https://cdn.pixabay.com/video/2021/08/31/87197-593755226_tiny.mp4"

# Abstract motion (Pixabay)
URL3="https://cdn.pixabay.com/video/2024/01/24/197780-906034997_tiny.mp4"

# Extract motion from any of these:
python make_warped_noise.py "$URL1" --output_folder test_motion
```

## Generate Video with Transferred Motion

After extracting motion, generate a new video:

```bash
conda activate flow

python cut_and_drag_inference.py dance_motion_test \
    --prompt "A cartoon robot performing choreography" \
    --output_mp4_path robot_dance.mp4 \
    --device cuda \
    --num_inference_steps 30
```

**Time:** ~5-15 minutes depending on GPU
**Output:** `robot_dance.mp4`

## Test Script

Run this complete test:

```bash
#!/bin/bash
# Complete motion transfer test

echo "=== Motion Transfer Test ==="

# Test 1: Use existing data
echo "Test 1: Using existing warped noise..."
conda activate flow
python cut_and_drag_inference.py my_warped_noise \
    --prompt "A dancing hologram" \
    --output_mp4_path test1_hologram.mp4 \
    --device cuda \
    --num_inference_steps 20

echo "✓ Test 1 complete: test1_hologram.mp4"

# Test 2: Extract from URL and generate
echo "Test 2: Full pipeline with new video..."
conda activate flow_warp
python make_warped_noise.py \
    "https://cdn.pixabay.com/video/2023/08/05/174963-851965097_tiny.mp4" \
    --output_folder test2_motion

conda activate flow
python cut_and_drag_inference.py test2_motion \
    --prompt "An oil painting in motion" \
    --output_mp4_path test2_painting.mp4 \
    --device cuda \
    --num_inference_steps 20

echo "✓ Test 2 complete: test2_painting.mp4"

echo "=== All tests complete! ==="
ls -lh test*.mp4
```

## Verify Results

After running, check your outputs:

```bash
# List generated videos
ls -lh *.mp4

# Check video info
ffmpeg -i test1_hologram.mp4 2>&1 | grep "Duration\|Stream"

# Play video (if you have a player)
vlc test1_hologram.mp4
# or
mpv test1_hologram.mp4
# or
ffplay test1_hologram.mp4
```

## Expected Results

**Input:** Your video motion pattern
**Output:** New 49-frame video where:
- Motion timing and dynamics match input
- Visual content matches your text prompt
- Resolution: 480x720 (CogVideoX default)
- Length: ~4 seconds at 12fps

## Troubleshooting

### "output_folder already exists"
```bash
rm -rf test_motion  # or use different name
python make_warped_noise.py video.mp4 --output_folder test_motion_v2
```

### CUDA out of memory
```bash
# Reduce inference steps
--num_inference_steps 15

# Or use different model
--model_name "T2V2B"  # Smaller 2B parameter model
```

### Video quality issues
```bash
# Increase inference steps for better quality
--num_inference_steps 50

# Adjust degradation for more/less motion
--degradation 0.0  # Full motion (recommended)
--degradation 0.3  # Slightly less motion
```

## Next Steps

After basic testing:

1. **Experiment with prompts:**
   - Try different artistic styles
   - Test with various subjects
   - Combine image + prompt (I2V mode)

2. **Control motion strength:**
   - Use `--degradation` parameter
   - 0.0 = full motion, 1.0 = no motion

3. **Advanced control:**
   - Use frequency decomposition (see QUICK_START_FREQUENCY.md)
   - Remove camera shake
   - Filter specific frequency bands

## Performance Notes

**make_warped_noise.py (flow_warp env):**
- Time: 1-3 minutes for 49 frames
- GPU: ~2-4GB for RAFT optical flow
- Output: ~8MB warped noise file

**cut_and_drag_inference.py (flow env):**
- Time: 5-15 minutes (depends on steps and GPU)
- GPU: ~8-16GB for CogVideoX-5B
- Output: ~2-5MB video file

## Success Criteria

Your test is successful if:
- ✓ Video generates without errors
- ✓ Output shows motion similar to input
- ✓ Visual content matches your prompt
- ✓ Video is smooth and coherent

## Common Test Prompts

Try these prompts to see different styles:

```bash
# Artistic styles
"A watercolor painting with flowing colors"
"An oil painting with visible brush strokes"
"A pencil sketch animation"
"A neon sign glowing in the dark"

# Characters/Objects
"A robot dancing with LED lights"
"A cartoon character performing ballet"
"A dancing skeleton made of flames"
"Colorful balloons floating and swaying"

# Abstract
"Flowing liquid metal"
"Dancing flames and embers"
"Swirling galaxies in space"
"Abstract geometric shapes morphing"

# Nature
"Autumn leaves falling and swirling"
"Ocean waves crashing dynamically"
"Northern lights dancing in the sky"
"Clouds forming and transforming"
```

## Ready to Test!

Run your first test now:

```bash
conda activate flow
python cut_and_drag_inference.py my_warped_noise \
    --prompt "A dancing flame" \
    --output_mp4_path my_first_test.mp4 \
    --device cuda \
    --num_inference_steps 20
```

This should take ~3-5 minutes and create `my_first_test.mp4`.

Good luck! 🎬

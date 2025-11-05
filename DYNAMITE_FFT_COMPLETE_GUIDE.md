# Complete Guide: Dynamite Video + FFT Motion Filtering

## Overview

This guide shows the complete workflow to:
1. Extract and visualize optical flow from dynamite video
2. Apply FFT filtering to separate motion frequencies
3. Create warped noise with filtered motion
4. Generate video using diffusion with motion control

**Video:** `test_video/dynamite_clip.mp4` (10 sec, 1280x720, 30fps)
**Goal:** Generate video with **hands-emphasized motion** (fast gestures amplified, slow drift removed)

---

## Part 1: Extract & Visualize Flow with FFT

### Step 1.1: Process Video with FFT Analysis

```bash
conda activate flow_warp

python process_dynamite_video.py
```

**What this does:**
- Loads dynamite video (301 frames → downsampled to 49 frames)
- Extracts optical flow using RAFT
- Decomposes flow into 4 frequency bands using FFT
- Analyzes motion spectrum
- Applies 4 filtering presets
- Creates visualizations (PNG + MP4)

**Output:** `dynamite_results/` folder containing:
```
dynamite_results/
├── original_flow.npy                    # Raw optical flow (48, 2, 480, 720)
├── original_flow.png                    # 5-frame visualization
├── original_flow.mp4                    # Full flow video
├── motion_analysis.json                 # Frequency analysis data
├── motion_spectrum.png                  # 4-panel analysis charts
│
├── hands_emphasized_flow.npy            # Filtered: [0.0, 0.3, 1.5, 2.0]
├── hands_emphasized_flow.png
├── hands_emphasized_flow.mp4
├── hands_emphasized_comparison.png      # Before/after comparison
│
├── ultra_smooth_flow.npy                # Filtered: [1.0, 0.8, 0.0, 0.0]
├── ultra_smooth_flow.png
├── ultra_smooth_flow.mp4
├── ultra_smooth_comparison.png
│
├── extreme_hands_flow.npy               # Filtered: [0.0, 0.0, 2.0, 3.0]
├── extreme_hands_flow.png
├── extreme_hands_flow.mp4
├── extreme_hands_comparison.png
│
├── remove_shake_flow.npy                # Filtered: [1.0, 1.0, 0.8, 0.0]
├── remove_shake_flow.png
├── remove_shake_flow.mp4
└── remove_shake_comparison.png
```

**Time:** ~2 minutes

**Check results:**
```bash
# View comparison images
xdg-open dynamite_results/hands_emphasized_comparison.png

# Check motion analysis
cat dynamite_results/motion_analysis.json

# View motion spectrum
xdg-open dynamite_results/motion_spectrum.png
```

### Step 1.2: Resize Flows to Latent Resolution

The flows from Step 1.1 are at full resolution (480x720), but the diffusion pipeline uses latent resolution (240x360). Resize them:

```bash
# Resize all 4 presets
for preset in hands_emphasized ultra_smooth extreme_hands remove_shake; do
    python resize_filtered_flow.py \
        dynamite_results/${preset}_flow.npy \
        240 360
done
```

**Output:** `*_flow_240x360.npy` files in `dynamite_results/`

**Time:** < 10 seconds

---

## Part 2: Create Warped Noise from Dynamite

### Step 2.1: Generate Warped Noise

```bash
conda activate flow_warp

python make_warped_noise.py \
    test_video/dynamite_clip.mp4 \
    dynamite_warped_noise
```

**What this does:**
- Preprocesses video (resize to 720x480, crop, 49 frames)
- Extracts optical flow at latent resolution (240x360)
- Warps noise using the flow
- Saves preprocessed video, flows, and warped noise

**Output:** `dynamite_warped_noise/` folder:
```
dynamite_warped_noise/
├── input.mp4                    # Preprocessed dynamite video
├── flows_dxdy.npy               # Optical flow (48, 2, 240, 360)
├── noises.npy                   # Warped noise (49, 60, 90, 16)
├── first_frame.png              # First frame preview
├── noise_video.mp4              # Noise visualization
├── visualization_video.mp4      # RGB + noise side-by-side
└── visualization_images/        # Per-frame visualizations
```

**Time:** ~2-3 minutes

**Verify:**
```bash
# Check flow shape
python -c "
import numpy as np
flow = np.load('dynamite_warped_noise/flows_dxdy.npy')
print(f'Flow shape: {flow.shape}')
print(f'Magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.3f}')
"
# Expected: Flow shape: (48, 2, 240, 360), Magnitude: ~1.056
```

### Step 2.2: Integrate FFT-Filtered Flow

Replace the flow in `dynamite_warped_noise/` with the FFT-filtered version:

```bash
# Choose a preset (hands_emphasized, ultra_smooth, extreme_hands, or remove_shake)
python integrate_filtered_flow.py integrate \
    dynamite_results/hands_emphasized_flow_240x360.npy \
    dynamite_warped_noise/
```

**What this does:**
- Backs up original flow to `flows_dxdy_original.npy`
- Replaces `flows_dxdy.npy` with filtered version
- Shows magnitude comparison

**Output:**
```
✓ Backed up original flow to: dynamite_warped_noise/flows_dxdy_original.npy
Original: 1.056 → Filtered: 1.400 (+32.6%)
```

**Time:** < 1 second

---

## Part 3: Generate Video with Diffusion

### Step 3.1: Generate with Default Settings

```bash
conda activate flow

python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_hands_emphasized.mp4 \
    --degradation 0.0 \
    --num_inference_steps 50
```

**Parameters:**
- `dynamite_warped_noise` - Input folder (warped noise + filtered flow)
- `dynamite_hands_emphasized.mp4` - Output video filename
- `--degradation 0.0` - Pure warped noise (maximum motion control)
- `--num_inference_steps 50` - Quality setting (higher = better)

**Time:** ~3-5 minutes (depends on GPU)

**Output:** `dynamite_hands_emphasized.mp4` (49 frames, 6s @ 8fps)

### Step 3.2: Generate with Text Prompt

```bash
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_hands_emphasized_with_prompt.mp4 \
    --degradation 0.0 \
    --prompt "A person dancing with energetic hand movements, vibrant colors, high quality" \
    --guidance_scale 6.0 \
    --num_inference_steps 50
```

**Additional parameters:**
- `--prompt` - Text description to guide generation
- `--guidance_scale 6.0` - How strongly to follow the prompt

### Step 3.3: Test Different Degradation Levels

```bash
# Maximum motion control (0.0) - follows flow exactly
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_deg0.0.mp4 \
    --degradation 0.0

# Medium control (0.3) - 70% flow, 30% random
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_deg0.3.mp4 \
    --degradation 0.3

# Light control (0.5) - 50% flow, 50% random
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_deg0.5.mp4 \
    --degradation 0.5
```

---

## Part 4: Compare All FFT Presets

Generate videos for all 4 filtering presets:

```bash
conda activate flow_warp

for preset in hands_emphasized ultra_smooth extreme_hands remove_shake; do
    echo "========================================="
    echo "Generating: ${preset}"
    echo "========================================="

    # Integrate filtered flow
    python integrate_filtered_flow.py integrate \
        dynamite_results/${preset}_flow_240x360.npy \
        dynamite_warped_noise/

    # Generate video
    conda activate flow
    python cut_and_drag_inference.py \
        dynamite_warped_noise \
        dynamite_${preset}.mp4 \
        --degradation 0.0 \
        --num_inference_steps 30

    conda activate flow_warp
done

echo "Done! Generated 4 videos:"
echo "  - dynamite_hands_emphasized.mp4   (+32% magnitude, fast motion amplified)"
echo "  - dynamite_ultra_smooth.mp4       (-28% magnitude, only slow motion)"
echo "  - dynamite_extreme_hands.mp4      (+86% magnitude, ONLY fast motion)"
echo "  - dynamite_remove_shake.mp4       (+1% magnitude, subtle shake removal)"
```

**Time:** ~15-20 minutes total

---

## Quick Reference: Copy-Paste Commands

### Full Pipeline (One Command Per Step)

```bash
# STEP 1: Extract and analyze flow
conda activate flow_warp
python process_dynamite_video.py

# STEP 2: Resize filtered flows
for preset in hands_emphasized ultra_smooth extreme_hands remove_shake; do
    python resize_filtered_flow.py dynamite_results/${preset}_flow.npy 240 360
done

# STEP 3: Create warped noise
python make_warped_noise.py test_video/dynamite_clip.mp4 dynamite_warped_noise

# STEP 4: Integrate filtered flow
python integrate_filtered_flow.py integrate \
    dynamite_results/hands_emphasized_flow_240x360.npy \
    dynamite_warped_noise/

# STEP 5: Generate video
conda activate flow
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_hands_emphasized.mp4 \
    --degradation 0.0 \
    --num_inference_steps 50
```

### Just Generate (If Already Processed)

```bash
# If you already have dynamite_warped_noise/ with filtered flow
conda activate flow

python cut_and_drag_inference.py \
    dynamite_warped_noise \
    output.mp4 \
    --degradation 0.0
```

---

## Understanding the Results

### Motion Analysis

From `dynamite_results/motion_analysis.json`:

| Band | Frequency | Original Mag | After Filtering | Preset |
|------|-----------|--------------|-----------------|--------|
| 0 | 0.1-0.68 Hz | 0.80 | 0.00 | Removed (×0.0) |
| 1 | 0.68-3.1 Hz | 1.14 | 0.34 | Reduced (×0.3) |
| 2 | 3.1-5.6 Hz | 1.22 | 1.83 | Amplified (×1.5) |
| 3 | 5.6-6.0 Hz | 1.01 | 2.02 | Doubled (×2.0) |

**Interpretation:**
- **Band 0-1 (slow):** Body drift and sway → **REMOVED/REDUCED**
- **Band 2-3 (fast):** Hand gestures and fast motion → **AMPLIFIED**

**Result:** Body appears more static, hand movements more prominent!

### Magnitude Changes

```
Original flow in warped noise:  1.056
After hands_emphasized filter:  1.400  (+32.6%)
After ultra_smooth filter:      0.763  (-27.7%)
After extreme_hands filter:     1.962  (+85.8%)
After remove_shake filter:      1.067  (+1.0%)
```

### Visual Comparison

Check the comparison images:
```bash
# View before/after for each preset
xdg-open dynamite_results/hands_emphasized_comparison.png
xdg-open dynamite_results/ultra_smooth_comparison.png
xdg-open dynamite_results/extreme_hands_comparison.png
xdg-open dynamite_results/remove_shake_comparison.png
```

**Color = Direction:**
- Red = Right, Yellow = Down-right, Green = Down
- Cyan = Left, Blue = Up-left, Magenta = Up

**Brightness = Speed:**
- Dark = Slow/no motion
- Bright = Fast motion

---

## Troubleshooting

### Error: Shape Mismatch

```
❌ Error: Shape mismatch!
   Original: (48, 2, 240, 360)
   Filtered: (48, 2, 480, 720)
```

**Solution:** Resize the filtered flow first:
```bash
python resize_filtered_flow.py \
    dynamite_results/hands_emphasized_flow.npy \
    240 360
```

### Error: CUDA Out of Memory

```bash
# Use low VRAM mode
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    output.mp4 \
    --low_vram True \
    --degradation 0.0
```

### Warning: Folder Already Exists

```
RuntimeError: The given output_folder='dynamite_warped_noise' already exists!
```

**Solution:** Delete or rename the existing folder:
```bash
rm -rf dynamite_warped_noise
# OR
mv dynamite_warped_noise dynamite_warped_noise_backup
```

---

## Advanced Usage

### Custom FFT Weights

Design your own weights based on the motion analysis:

```python
# Create custom filtering script
cat > apply_custom_fft.py << 'EOF'
import numpy as np
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

# Load flow from warped noise
flow = np.load('dynamite_warped_noise/flows_dxdy.npy')
print(f'Original magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.3f}')

# Apply FFT
editor = FrequencyMotionEditor(flow, fps=12)
editor.decompose_frequencies(num_bands=4)

# Custom weights: emphasize band 2 only
weights = [0.5, 1.0, 2.5, 0.5]
filtered = editor.reconstruct_selective(weights)

print(f'Filtered magnitude: {np.sqrt(filtered[:,0]**2 + filtered[:,1]**2).mean():.3f}')

# Backup and save
np.save('dynamite_warped_noise/flows_dxdy_original.npy', flow)
np.save('dynamite_warped_noise/flows_dxdy.npy', filtered)
print('✓ Applied custom FFT filtering')
EOF

conda activate flow_warp
python apply_custom_fft.py
```

### Process Different Video

```bash
# Replace test_video/dynamite_clip.mp4 with your video
python make_warped_noise.py \
    /path/to/your/video.mp4 \
    your_video_warped_noise
```

### Use Different Model

```bash
# Use CogVideoX-2B (faster, lower quality)
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    output.mp4 \
    --model_name I2V2B \
    --degradation 0.0

# Use Text-to-Video model
python cut_and_drag_inference.py \
    dynamite_warped_noise \
    output.mp4 \
    --model_name T2V5B \
    --prompt "A person dancing" \
    --degradation 0.0
```

---

## Summary

**What You Have Now:**

✅ **Flow Analysis:** `dynamite_results/` with visualizations
✅ **Filtered Flows:** 4 presets at both resolutions
✅ **Warped Noise:** `dynamite_warped_noise/` with filtered flow integrated
✅ **Ready to Generate:** Everything set up for video diffusion

**Next Command to Run:**

```bash
conda activate flow

python cut_and_drag_inference.py \
    dynamite_warped_noise \
    dynamite_hands_emphasized.mp4 \
    --degradation 0.0 \
    --num_inference_steps 50
```

**Expected Result:**

A video where:
- Slow body drift is removed (bands 0-1 filtered out)
- Fast hand gestures are amplified (bands 2-3 boosted)
- Motion appears more dynamic and hand-focused

**Time to result:** ~5 minutes total (after setup)

---

## Files Created

```
.
├── test_video/
│   └── dynamite_clip.mp4              # Source video
│
├── dynamite_results/                   # FFT analysis and visualizations
│   ├── original_flow.npy              # (48, 2, 480, 720)
│   ├── *_flow.npy                     # 4 filtered versions (480x720)
│   ├── *_flow_240x360.npy             # 4 downsampled versions (240x360)
│   ├── *_comparison.png               # Before/after images
│   ├── motion_analysis.json           # Frequency analysis
│   ├── motion_spectrum.png            # Analysis charts
│   └── RESULTS_SUMMARY.md             # Detailed results
│
├── dynamite_warped_noise/             # Ready for diffusion
│   ├── input.mp4                      # Preprocessed video
│   ├── flows_dxdy.npy                 # FFT-filtered flow (240x360)
│   ├── flows_dxdy_original.npy        # Backup of original flow
│   ├── noises.npy                     # Warped noise (16 channels)
│   └── visualization_video.mp4        # Preview
│
└── dynamite_hands_emphasized.mp4      # Generated video (output)
```

🎬 **You're all set! Run the generation command and enjoy your motion-controlled video!**

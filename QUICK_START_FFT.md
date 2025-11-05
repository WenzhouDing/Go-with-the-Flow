# Quick Start: FFT Motion Filtering

## Complete Workflow (Dynamite Example)

### Step 1: Process Video with FFT
```bash
conda activate flow_warp

# Process video and generate all visualizations
python process_dynamite_video.py
```

**Output:** `dynamite_results/` folder with:
- 4 filtered flow versions (hands emphasized, ultra smooth, extreme hands, remove shake)
- Before/after comparison images
- Motion spectrum analysis
- Flow visualization videos

### Step 2: Resize to Match Your Pipeline
```bash
# If you get a shape mismatch, resize the filtered flow
python resize_filtered_flow.py \
    dynamite_results/hands_emphasized_flow.npy \
    240 360
```

**When to resize:**
- Your existing flows are at latent resolution (e.g., 240x360)
- Dynamite flows are at full resolution (480x720)
- You see "Shape mismatch" error

### Step 3: Integrate Filtered Flow
```bash
# Replace flows in your warped noise folder
python integrate_filtered_flow.py integrate \
    dynamite_results/hands_emphasized_flow_240x360.npy \
    my_warped_noise/
```

**Result:**
- Original flow backed up to `flows_dxdy_original.npy`
- Filtered flow now in `flows_dxdy.npy`
- Ready for video generation

### Step 4: Generate Video
```bash
conda activate flow

python cut_and_drag_inference.py \
    --folder_path my_warped_noise \
    --degradation 0.0 \
    --output_name dynamite_hands_emphasized.mp4
```

## Available Presets

### 1. Hands Emphasized `[0.0, 0.3, 1.5, 2.0]`
- **Effect:** Amplify fast motion, remove slow drift
- **Magnitude change:** +28% (before resize), +152% (after resize to your pipeline)
- **Use for:** Sign language, hand gestures, conductor movements

### 2. Ultra Smooth `[1.0, 0.8, 0.0, 0.0]`
- **Effect:** Keep only slow motion, remove all fast movements
- **Magnitude change:** -30%
- **Use for:** Smooth dance, flowing movements, stabilized footage

### 3. Extreme Hands `[0.0, 0.0, 2.0, 3.0]`
- **Effect:** ONLY fast motion, body completely static
- **Magnitude change:** +80%
- **Use for:** Isolating hand/finger movements, drumming, rapid gestures

### 4. Remove Shake `[1.0, 1.0, 0.8, 0.0]`
- **Effect:** Subtle shake removal, preserve content motion
- **Magnitude change:** -2%
- **Use for:** Natural stabilization, removing high-frequency jitter

## Try Different Presets

```bash
# Integrate different presets to compare
for preset in hands_emphasized ultra_smooth extreme_hands remove_shake; do
    echo "Testing ${preset}..."

    # Integrate
    python integrate_filtered_flow.py integrate \
        dynamite_results/${preset}_flow_240x360.npy \
        my_warped_noise/

    # Generate
    conda activate flow
    python cut_and_drag_inference.py \
        --folder_path my_warped_noise \
        --degradation 0.0 \
        --output_name dynamite_${preset}.mp4
done
```

## Understanding the Visualizations

### Color = Direction
- **Red:** Right
- **Yellow:** Down-right
- **Green:** Down
- **Cyan:** Left
- **Blue:** Up-left
- **Magenta:** Up

### Brightness = Speed
- **Dark:** Little/no motion
- **Bright:** Fast motion

### Files to Check
1. `*_comparison.png` - Side-by-side before/after
2. `motion_spectrum.png` - Frequency analysis charts
3. `*_flow.mp4` - Full flow videos

## Your Dynamite Results

**Original video:** `test_video/dynamite_clip.mp4`
- 301 frames, 1280x720, 30fps

**Processed at:** 480x720, 12fps, 48 frames
**Original magnitude:** 2.19

**After resize to 240x360:**
- hands_emphasized: 1.400 (+152% vs your original 0.556)
- ultra_smooth: 0.763 (+37% vs your original)
- extreme_hands: 1.962 (+253% vs your original)
- remove_shake: 1.067 (+92% vs your original)

## Troubleshooting

### Shape Mismatch Error
**Problem:** Filtered flow resolution doesn't match target
**Solution:** Use `resize_filtered_flow.py` to downsample

```bash
python resize_filtered_flow.py \
    dynamite_results/hands_emphasized_flow.npy \
    <target_height> <target_width>
```

### How to Check Target Resolution
```bash
# Check your warped noise folder
python -c "
import numpy as np
flow = np.load('my_warped_noise/flows_dxdy.npy')
print(f'Target resolution: {flow.shape}')
print(f'Height: {flow.shape[2]}, Width: {flow.shape[3]}')
"
```

### Magnitude Too High/Low
**Problem:** Generated video has too much/little motion
**Solution:** Adjust degradation parameter or use different preset

```bash
# More motion (less degradation)
--degradation 0.0

# Less motion (more degradation)
--degradation 0.3
```

## Custom Weights

Design your own based on the motion analysis:

```python
# From dynamite_results/motion_analysis.json:
# Band 0 (0.1-0.68Hz): 0.80 magnitude
# Band 1 (0.68-3.1Hz): 1.14 magnitude
# Band 2 (3.1-5.6Hz): 1.22 magnitude
# Band 3 (5.6-6.0Hz): 1.01 magnitude

# Example: Emphasize band 2 only
custom = [0.5, 1.0, 2.0, 0.5]

# Apply custom weights (need to modify process_dynamite_video.py)
# Or use frequency_motion_transfer.py directly
```

## Next Steps

1. ✅ **Done:** Processed dynamite video with FFT
2. ✅ **Done:** Resized flows to match your pipeline
3. ✅ **Done:** Integrated hands-emphasized flow
4. **TODO:** Generate video and see the results!

```bash
conda activate flow
python cut_and_drag_inference.py \
    --folder_path my_warped_noise \
    --degradation 0.0 \
    --output_name dynamite_hands_emphasized.mp4
```

5. **Next:** Try other presets and compare!

## Files Reference

**Scripts:**
- `process_dynamite_video.py` - Complete FFT pipeline
- `resize_filtered_flow.py` - Resize flows to match resolution
- `integrate_filtered_flow.py` - Replace flows in warped noise
- `test_hand_emphasis.py` - Test FFT on existing flows

**Results:**
- `dynamite_results/` - All processed flows and visualizations
- `my_warped_noise/` - Your warped noise (now with filtered flow!)

**Docs:**
- `docs/FLOW_VISUALIZATION_GUIDE.md` - How to read visualizations
- `docs/FFT_HAND_BODY_DEMO.md` - Hand vs body motion control demo
- `docs/FFT_ADVANCED_USE_CASES.md` - Advanced techniques
- `dynamite_results/RESULTS_SUMMARY.md` - Detailed results analysis

## Summary

You now have:
- ✅ FFT-filtered optical flows (4 presets)
- ✅ Visual comparisons showing before/after
- ✅ Resized flows matching your pipeline resolution
- ✅ Integrated into `my_warped_noise/` folder
- ✅ Ready to generate motion-controlled videos!

**Your next command:**
```bash
conda activate flow
python cut_and_drag_inference.py \
    --folder_path my_warped_noise \
    --degradation 0.0 \
    --output_name dynamite_hands_emphasized.mp4
```

🎬 Happy motion controlling!

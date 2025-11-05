# Complete FFT-Based Motion Filtering Demo

## What You Have Now

✓ **Frequency-decomposed motion transfer system** - Fully implemented and tested (14/14 tests passing)
✓ **Your test video** - `my_warped_noise.mp4` (4 seconds, 12fps)
✓ **Ready-to-run scripts** - Extract, analyze, filter, visualize

## The Point of Using FFT

FFT (Fast Fourier Transform) lets you:
1. **Decompose motion into frequency bands** - Separate slow movements from fast shakes
2. **Selectively filter motion** - Remove camera shake while keeping choreography
3. **Analyze motion patterns** - Understand what type of motion is in each frequency range
4. **Control motion transfer** - Fine-tune which motion frequencies to keep or remove

## Complete Demo Workflow

### STEP 1: Extract Optical Flow from Your Video

```bash
conda activate flow_warp

# Extract flow from your video
python extract_flow_sequence.py my_warped_noise.mp4 \
    --output extracted_flows.npy \
    --fps 12
```

**Time:** ~30-60 seconds
**Output:** `extracted_flows.npy` - Raw optical flow (shape: 48, 2, 480, 720)

**What it does:**
- Loads video (resizes to 480x720, 49 frames)
- Extracts optical flow using RAFT
- Saves complete flow sequence

### STEP 2: Analyze Motion with FFT

```bash
# See what motion is in each frequency band
python frequency_motion_transfer.py my_warped_noise.mp4 \
    --flow_path extracted_flows.npy \
    --output_dir motion_analysis \
    --analyze_only \
    --num_bands 4 \
    --fps 12
```

**Output:**
```
==================================================
FREQUENCY BAND ANALYSIS REPORT
==================================================

--- Band 0 ---
  Mean Magnitude:      X.XX
  Temporal Variance:   X.XXXX
  Spatial Variance:    X.XXXX

  Classification Scores:
    Camera Shake:      0.XXX
    Body Movement:     0.XXX
    Micro Motion:      0.XXX
  Dominant Type: BODY_MOVEMENT

--- Band 1 ---
  ...
```

**What this shows:**
- Which bands contain camera shake
- Which bands have body/object movement
- Motion statistics for each frequency range

### STEP 3: Apply FFT Filtering (Remove Shake)

```bash
# Automatically remove camera shake
python frequency_motion_transfer.py my_warped_noise.mp4 \
    --flow_path extracted_flows.npy \
    --output_dir shake_removed \
    --remove_shake \
    --num_bands 4 \
    --fps 12
```

**Output files:**
- `shake_removed/raw_flows.npy` - Original flow
- `shake_removed/clean_flow.npy` - Filtered flow (shake removed!)
- `shake_removed/band_0_flow.npy` - Low frequency band
- `shake_removed/band_1_flow.npy` - Mid-low frequency
- `shake_removed/band_2_flow.npy` - Mid-high frequency
- `shake_removed/band_3_flow.npy` - High frequency (shake)
- `shake_removed/motion_analysis.json` - Detailed analysis

**Motion reduction stats:**
```
Original mean magnitude:  XX.XX
Cleaned mean magnitude:   XX.XX
Motion reduction:         XX.X%
```

### STEP 4: Visualize Frequency Decomposition

```bash
# Create visualization plots
python test_frequency_decomposition.py shake_removed/ \
    --output visualizations/
```

**Generated plots:**
1. `frequency_bands_analysis.png` - Energy over time and space
2. `band_statistics.png` - Comparison of band characteristics
3. `motion_classification.png` - Motion type scores per band

### STEP 5: Custom Frequency Filtering

```bash
# Example: Keep smooth motion, remove fast movements
python frequency_motion_transfer.py my_warped_noise.mp4 \
    --flow_path extracted_flows.npy \
    --output_dir smooth_motion \
    --custom_weights 1.0,1.0,0.2,0.0 \
    --num_bands 4 \
    --fps 12
```

**Band weights explained:**
- `1.0` = Keep 100% of this frequency band
- `0.5` = Keep 50% (blend with zero)
- `0.0` = Remove completely

**Example configurations:**

```bash
# Remove all high frequency (smooth only)
--custom_weights 1.0,1.0,0.0,0.0

# Keep everything except highest band
--custom_weights 1.0,1.0,0.8,0.0

# Emphasize mid frequencies
--custom_weights 0.5,1.0,1.0,0.2

# Remove shake and micro-motion
--custom_weights 1.0,1.0,0.5,0.0
```

## Understanding Frequency Bands (12fps video)

For your 12fps video, the Nyquist frequency is 6 Hz:

| Band | Frequency Range | Typical Content | Recommendation |
|------|----------------|-----------------|----------------|
| 0 | 0.1 - 0.68 Hz | Camera pans, slow drift | Usually keep (1.0) |
| 1 | 0.68 - 3.1 Hz | Body movements, choreography | Keep (1.0) |
| 2 | 3.1 - 5.6 Hz | Fast gestures, quick movements | Partial (0.5-0.8) |
| 3 | 5.6 - 6.0 Hz | Camera shake, noise | Remove (0.0) |

## One-Line Demo Commands

### Quick Analysis:
```bash
conda activate flow_warp && python extract_flow_sequence.py my_warped_noise.mp4 --output flows.npy --fps 12 && python frequency_motion_transfer.py my_warped_noise.mp4 --flow_path flows.npy --output_dir demo --analyze_only --fps 12
```

### Extract + Filter + Visualize:
```bash
conda activate flow_warp && python extract_flow_sequence.py my_warped_noise.mp4 --output flows.npy --fps 12 && python frequency_motion_transfer.py my_warped_noise.mp4 --flow_path flows.npy --output_dir filtered --remove_shake --fps 12 && python test_frequency_decomposition.py filtered/
```

## Automated Demo Script

Run the complete pipeline automatically:

```bash
./run_fft_demo.sh
```

This runs all steps and creates a timestamped output folder with:
- Extracted flows
- Frequency decomposition
- Filtered flows
- Analysis reports
- Visualization plots

## What Makes This Different

**Traditional Motion Transfer:**
```
Video → Extract All Motion → Transfer Everything
```
- Camera shake transfers too
- No control over motion types
- Can't separate different motion components

**FFT-Based Motion Transfer (What We Built):**
```
Video → Extract Motion → FFT Decompose → Selective Filter → Clean Motion
                              ↓
                    [4 Frequency Bands]
                    Band 0: Slow (keep)
                    Band 1: Medium (keep)
                    Band 2: Fast (partial)
                    Band 3: Shake (remove)
```
- Remove unwanted motion (shake, jitter)
- Keep desired motion (choreography, main movements)
- Fine-grained control per frequency band
- Understand motion composition

## Practical Examples

### Example 1: Remove Camera Shake from Handheld Video
```bash
python frequency_motion_transfer.py shaky_dance.mp4 \
    --output_dir stabilized \
    --remove_shake \
    --fps 30
```
**Result:** Choreography preserved, shake removed

### Example 2: Keep Only Smooth Motion
```bash
python frequency_motion_transfer.py dance.mp4 \
    --output_dir smooth \
    --custom_weights 1.0,0.8,0.0,0.0 \
    --fps 30
```
**Result:** Smooth, flowing motion only

### Example 3: Analyze Before Deciding
```bash
# Step 1: Analyze
python frequency_motion_transfer.py video.mp4 \
    --output_dir analysis \
    --analyze_only

# Step 2: Check the report
cat analysis/motion_analysis.json

# Step 3: Apply custom weights based on analysis
python frequency_motion_transfer.py video.mp4 \
    --output_dir filtered \
    --custom_weights <based_on_analysis>
```

## Integration with Full Pipeline

After FFT filtering, use clean flow for video generation:

```bash
# 1. Extract and filter motion
conda activate flow_warp
python extract_flow_sequence.py source.mp4 --output flows.npy
python frequency_motion_transfer.py source.mp4 \
    --flow_path flows.npy \
    --output_dir filtered \
    --remove_shake

# 2. TODO: Integrate with make_warped_noise.py
# (Manual integration currently - use filtered/clean_flow.npy)

# 3. Generate video
conda activate flow
python cut_and_drag_inference.py noise_folder \
    --prompt "Your prompt" \
    --output_mp4_path result.mp4
```

## Verification

Check that FFT filtering worked:

```bash
# 1. Compare original vs filtered flow
python -c "
import numpy as np
raw = np.load('filtered/raw_flows.npy')
clean = np.load('filtered/clean_flow.npy')
print(f'Original magnitude: {np.sqrt(raw[:,0]**2 + raw[:,1]**2).mean():.2f}')
print(f'Filtered magnitude: {np.sqrt(clean[:,0]**2 + clean[:,1]**2).mean():.2f}')
print(f'Reduction: {(1 - np.sqrt(clean[:,0]**2 + clean[:,1]**2).mean() / np.sqrt(raw[:,0]**2 + raw[:,1]**2).mean()) * 100:.1f}%')
"

# 2. View visualizations
ls -lh visualizations/*.png
```

## Success Criteria

✓ Flow extraction completes without errors
✓ FFT decomposition creates 4 bands
✓ Motion analysis shows sensible classifications
✓ Filtered flow has reduced magnitude (shake removed)
✓ Visualizations show clear frequency separation

## Next Steps

1. **Run the demo:** `./run_fft_demo.sh`
2. **Experiment with band weights:** Try different combinations
3. **Test with different videos:** Various motion types
4. **Integrate with generation:** Use clean flow for final videos

## Why This Matters

Traditional motion transfer is "all or nothing" - you get all the motion including unwanted shake and jitter.

With FFT-based filtering, you can:
- Transfer smooth choreography without camera shake
- Remove high-frequency noise while keeping main movements
- Understand motion composition before transfer
- Create cleaner, more controlled motion transfer

This is especially useful for:
- Handheld/action cam footage (remove shake)
- Professional motion capture (isolate specific movements)
- Artistic control (choose motion characteristics)
- Quality improvement (remove artifacts)

## Ready to Try!

Your video is ready. Run this now:

```bash
./run_fft_demo.sh
```

Or step by step:

```bash
conda activate flow_warp
python extract_flow_sequence.py my_warped_noise.mp4 --output flows.npy --fps 12
python frequency_motion_transfer.py my_warped_noise.mp4 --flow_path flows.npy --output_dir demo --remove_shake --fps 12
python test_frequency_decomposition.py demo/
```

The magic happens in the FFT decomposition - it separates your motion into frequency components so you can keep what you want and remove what you don't!

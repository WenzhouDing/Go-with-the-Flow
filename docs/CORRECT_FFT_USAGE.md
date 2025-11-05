# Correct FFT Usage - Understanding the Pipeline

## The Pipeline (Correct Understanding)

```
SOURCE VIDEO (real motion)
    ↓
make_warped_noise.py
    ↓
OUTPUT FOLDER/
├── input.mp4              ← Preprocessed source (720x480, 49 frames)
├── flows_dxdy.npy         ← Optical flow (THIS is what FFT uses!)
├── noises.npy             ← Warped noise (for diffusion)
└── visualization_video.mp4 ← RGB + noise side-by-side (NOT for FFT input!)
```

## Common Mistake

❌ **WRONG:** Using `visualization_video.mp4` as FFT input
- This is a visualization showing RGB + noise side by side
- NOT a source video!

✓ **CORRECT:** Either:
1. Use `input.mp4` (preprocessed source) for new extraction
2. Use existing `flows_dxdy.npy` directly
3. Use a completely new source video

## What Videos Work Best for FFT?

### IDEAL for FFT Filtering:

**1. Handheld Dance/Performance Videos**
- Has camera shake (high frequency)
- Has choreography/movement (low-mid frequency)
- FFT can separate and remove shake while keeping dance

**2. Action Camera Footage**
- GoPro, phone videos with movement
- Has jitter and shake from motion
- Has main subject motion you want to keep

**3. Phone Videos**
- Unstabilized phone recordings
- Mix of camera shake + subject motion

### NOT Useful for FFT:

**1. Tripod/Stabilized Videos**
- Already smooth, no shake to remove
- Only one motion frequency

**2. Static Camera**
- No camera motion
- FFT won't help much

**3. Already Processed Videos**
- Pre-stabilized footage
- Nothing to filter out

## Your Current Data

You have: `my_warped_noise/`
- `input.mp4` - Your preprocessed source video (720x480, 4 sec)
- `flows_dxdy.npy` - Already extracted flows!

## Correct Usage

### Option 1: Use Existing Flow Data (Fastest)

Since you already have `flows_dxdy.npy`, use it directly:

```bash
conda activate flow_warp

# Load the existing flow and apply FFT filtering
python -c "
import numpy as np
from CommonSource.frequency_motion_editor import FrequencyMotionEditor
from CommonSource.motion_classifier import MotionClassifier

# Load existing flow
flow = np.load('my_warped_noise/flows_dxdy.npy')
print(f'Flow shape: {flow.shape}')

# Apply FFT decomposition
editor = FrequencyMotionEditor(flow, fps=12)
bands = editor.decompose_frequencies(num_bands=4)

# Analyze
analysis = editor.analyze_motion_spectrum()
MotionClassifier.print_analysis_report(analysis)

# Get recommended weights
weights = MotionClassifier.recommend_band_weights(analysis, remove_shake=True)
print(f'\nRecommended weights: {weights}')

# Apply filtering
clean_flow = editor.reconstruct_selective(weights)

# Save
np.save('fft_filtered_flow.npy', clean_flow)
print(f'\nFiltered flow saved: fft_filtered_flow.npy')
print(f'Original magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.2f}')
print(f'Filtered magnitude: {np.sqrt(clean_flow[:,0]**2 + clean_flow[:,1]**2).mean():.2f}')
"
```

### Option 2: Use the Pipeline with input.mp4

```bash
conda activate flow_warp

# Use the preprocessed source video
python frequency_motion_transfer.py my_warped_noise/input.mp4 \
    --output_dir fft_analysis \
    --analyze_only \
    --num_bands 4 \
    --fps 12

# Then apply filtering
python frequency_motion_transfer.py my_warped_noise/input.mp4 \
    --output_dir fft_filtered \
    --remove_shake \
    --num_bands 4 \
    --fps 12
```

### Option 3: Get a Better Test Video

For a clear FFT demo, download a handheld dance video:

```bash
conda activate flow_warp

# Example: Use a public domain handheld dance video
# (Replace URL with actual video)
python make_warped_noise.py "<DANCE_VIDEO_URL>" \
    --output_folder handheld_dance

# Then apply FFT
python frequency_motion_transfer.py handheld_dance/input.mp4 \
    --output_dir dance_filtered \
    --remove_shake \
    --fps 30
```

## Understanding Your Video Type

To check if FFT will be useful for your video:

```python
import numpy as np
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

# Load your flow
flow = np.load('my_warped_noise/flows_dxdy.npy')

# Decompose and analyze
editor = FrequencyMotionEditor(flow, fps=12)
editor.decompose_frequencies(num_bands=4)
analysis = editor.analyze_motion_spectrum()

# Check each band
for i in range(4):
    stats = analysis[f'band_{i}']
    print(f"\nBand {i}:")
    print(f"  Magnitude: {stats['mean_magnitude']:.2f}")
    print(f"  Temporal variance: {stats['temporal_variance']:.4f}")
    print(f"  Spatial variance: {stats['spatial_variance']:.4f}")
```

**What to look for:**
- **High temporal variance in high bands** = Camera shake present
- **High spatial variance in low bands** = Subject movement present
- If all bands have similar characteristics = One motion type (FFT less useful)

## Frequency Bands for 12fps Video

Your video is 12fps, so Nyquist frequency = 6 Hz:

| Band | Frequency | Typical Content |
|------|-----------|----------------|
| 0 | 0.1 - 0.68 Hz | Very slow movements, drifting |
| 1 | 0.68 - 3.1 Hz | Main body movements |
| 2 | 3.1 - 5.6 Hz | Fast movements, gestures |
| 3 | 5.6 - 6.0 Hz | Shake, jitter, noise |

## Real-World Example

**Before FFT:**
```
Video: Handheld phone recording of dancer
- Band 0 (0.1-0.68Hz): Slow body sway ✓ Keep
- Band 1 (0.68-3.1Hz): Dance moves ✓ Keep
- Band 2 (3.1-5.6Hz): Fast hand gestures ✓ Keep
- Band 3 (5.6-6Hz): Phone shake ✗ Remove
```

**After FFT:**
- Smooth dance motion preserved
- Camera shake removed
- Better motion transfer quality

## When FFT Doesn't Help

If your video is:
1. **Already smooth** - Nothing to filter
2. **Tripod shot** - No shake
3. **Only fast motion** - Can't separate components

Then regular motion transfer works fine - FFT is optional.

## Quick Check

```bash
# Check what's in your flow
conda activate flow_warp
python -c "
import numpy as np
flow = np.load('my_warped_noise/flows_dxdy.npy')
print(f'Shape: {flow.shape}')
print(f'Mean magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.2f}')
print(f'Max magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).max():.2f}')
print(f'Std magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).std():.2f}')
"
```

**What it means:**
- High std/mean ratio → Variable motion (good for FFT)
- Low std/mean ratio → Consistent motion (FFT less useful)

## Summary

✓ **DO use FFT for:**
- Handheld videos with shake
- Mixed frequency motion
- Videos where you want to remove specific motion types

✗ **DON'T use FFT for:**
- Already smooth videos
- Single motion type
- Tripod shots

✓ **For your case:**
Use the existing `flows_dxdy.npy` and see what the analysis shows!

# Your Video Analysis - Fast Hands vs Slow Body

## Your Motion Analysis Results

From `fft_filtered/motion_analysis.json`:

| Band | Frequency | Mean Magnitude | Interpretation |
|------|-----------|----------------|----------------|
| 0 | 0.1-0.68 Hz | **0.58** | ⭐ **DOMINANT** - Slow drift/sway |
| 1 | 0.68-3.1 Hz | **0.54** | ⭐ **DOMINANT** - Medium movements |
| 2 | 3.1-5.6 Hz | 0.17 | Minor - Fast gestures |
| 3 | 5.6-6 Hz | 0.09 | Minimal - Very fast/shake |

## What This Tells Us

**Your video has:**
- ✓ Strong low-frequency motion (bands 0-1) → Body movements, slow gestures
- ✓ Weak high-frequency motion (bands 2-3) → Not much fast hand motion

**This means:**
- Most motion is slow/medium speed (body swaying, slow movements)
- Less fast hand gestures in this particular video
- Good candidate for smoothing/stabilization

## Use Case: Emphasize Fast Hands (If They Exist)

Even though your video has mostly slow motion, if there ARE fast hand movements, you can amplify them:

### Example 1: Amplify Any Fast Motion (Hands)

```bash
# Remove slow drift, emphasize fast movements
python frequency_motion_transfer.py your_video.mp4 \
    --custom_weights 0.0,0.3,1.5,2.0 \
    --output_dir hands_emphasized \
    --fps 12
```

**What this does:**
- Band 0 (0.58 mag) → 0.0 weight = **REMOVE** body drift
- Band 1 (0.54 mag) → 0.3 weight = **REDUCE** slow movement
- Band 2 (0.17 mag) → 1.5 weight = **AMPLIFY 50%** fast gestures
- Band 3 (0.09 mag) → 2.0 weight = **DOUBLE** very fast motion

**Result:**
- Original: Band 2 has magnitude 0.17
- After: Band 2 magnitude becomes 0.17 × 1.5 = 0.26
- Body drift removed, fast motion emphasized

### Example 2: Super Smooth (Remove All Fast Motion)

```bash
# Keep only slow, smooth motion
python frequency_motion_transfer.py your_video.mp4 \
    --custom_weights 1.0,0.8,0.0,0.0 \
    --output_dir ultra_smooth \
    --fps 12
```

**Result:**
- Only bands 0-1 (slow motion) kept
- All fast motion removed
- Very smooth, flowing motion

### Example 3: Exaggerate Everything Fast

```bash
# Make fast motion VERY prominent
python frequency_motion_transfer.py your_video.mp4 \
    --custom_weights 0.1,0.5,2.0,3.0 \
    --output_dir fast_exaggerated \
    --fps 12
```

**Result:**
- Band 2: 0.17 × 2.0 = 0.34 (doubled!)
- Band 3: 0.09 × 3.0 = 0.27 (tripled!)
- Fast motion becomes more prominent than original slow motion

## For a REAL "Fast Hands, Slow Body" Video

If you get a video where:
- Hands gesture rapidly (high frequency)
- Body barely moves (low frequency)

The analysis would look like:

| Band | Magnitude | Content |
|------|-----------|---------|
| 0 | 0.1 | Almost no body drift |
| 1 | 0.3 | Minimal body sway |
| 2 | **1.5** | ⭐ Strong hand gestures |
| 3 | **2.0** | ⭐ Very fast finger movements |

Then you'd use:
```bash
# Keep hands, remove body
--custom_weights 0.0,0.0,1.0,1.2
```

## Quick Experiment for Your Video

Try these three versions to see the difference:

```bash
conda activate flow_warp

# Version 1: Original (all weights = 1.0)
python frequency_motion_transfer.py fft_filtered/raw_flows.npy \
    --custom_weights 1.0,1.0,1.0,1.0 \
    --output_dir v1_original

# Version 2: Smooth (remove fast)
python frequency_motion_transfer.py fft_filtered/raw_flows.npy \
    --custom_weights 1.0,0.8,0.0,0.0 \
    --output_dir v2_smooth

# Version 3: Emphasized fast (amplify bands 2-3)
python frequency_motion_transfer.py fft_filtered/raw_flows.npy \
    --custom_weights 0.2,0.5,2.0,3.0 \
    --output_dir v3_fast_emphasized
```

Then compare:
```python
import numpy as np

v1 = np.load('v1_original/clean_flow.npy')
v2 = np.load('v2_smooth/clean_flow.npy')
v3 = np.load('v3_fast_emphasized/clean_flow.npy')

print(f"V1 (original):  {np.sqrt(v1[:,0]**2 + v1[:,1]**2).mean():.2f}")
print(f"V2 (smooth):    {np.sqrt(v2[:,0]**2 + v2[:,1]**2).mean():.2f}")
print(f"V3 (fast emphasis): {np.sqrt(v3[:,0]**2 + v3[:,1]**2).mean():.2f}")
```

## Real World Example Script

Here's a complete script to test hand emphasis:

```python
import numpy as np
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

# Load your flow
flow = np.load('fft_filtered/raw_flows.npy')
print(f"Original flow magnitude: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).mean():.2f}")

# Create editor
editor = FrequencyMotionEditor(flow, fps=12)
editor.decompose_frequencies(num_bands=4)

# Test 1: Emphasize fast motion (hands)
print("\n=== Test 1: Emphasize Fast Motion ===")
weights_fast = [0.0, 0.3, 1.5, 2.0]
result_fast = editor.reconstruct_selective(weights_fast)
print(f"Emphasized fast: {np.sqrt(result_fast[:,0]**2 + result_fast[:,1]**2).mean():.2f}")

# Show per-band contribution
for i, w in enumerate(weights_fast):
    band_mag = np.sqrt(editor.freq_bands[i][:,0]**2 + editor.freq_bands[i][:,1]**2).mean()
    print(f"  Band {i}: {band_mag:.2f} × {w} = {band_mag * w:.2f}")

# Test 2: Smooth everything
print("\n=== Test 2: Ultra Smooth ===")
weights_smooth = [1.0, 0.8, 0.0, 0.0]
result_smooth = editor.reconstruct_selective(weights_smooth)
print(f"Ultra smooth: {np.sqrt(result_smooth[:,0]**2 + result_smooth[:,1]**2).mean():.2f}")

# Save both
np.save('hands_emphasized.npy', result_fast)
np.save('ultra_smooth.npy', result_smooth)
print("\n✓ Saved both versions")
```

## The Answer to Your Question

**Q: Can we use FFT to control hand motion (fast) vs body motion (slow)?**

**A: YES! Absolutely!**

**How:**
1. **Fast hands** → High frequency bands (2-3)
2. **Slow body** → Low frequency bands (0-1)
3. **Control:** Set different weights per band

**Your specific video:**
- Mostly slow motion (bands 0-1 dominant)
- To emphasize any fast motion: Use weights like `[0.0, 0.3, 1.5, 2.0]`
- To smooth everything: Use weights like `[1.0, 0.8, 0.0, 0.0]`

**For a REAL fast-hands video:**
- Use weights like `[0.0, 0.0, 1.0, 1.5]` to keep ONLY fast motion
- Body will appear static, hands will move

## Advanced: Spatial + Temporal Control

If you want even MORE control (different weights for hands region vs body region):

```python
# Create hand mask (e.g., upper part of frame)
mask = np.zeros((480, 720))
mask[:240, :] = 1.0  # Top half = hands

# Apply different weights to different regions
result = editor.apply_spatial_frequency_edit(
    mask=mask,
    band_weights_fg=[0.0, 0.5, 1.5, 2.0],  # Hands: emphasize fast
    band_weights_bg=[1.0, 0.8, 0.2, 0.0]   # Body: smooth
)
```

This gives you **pixel-level control** over which frequencies to keep!

## Summary

✅ **YES, FFT can control fast hands vs slow body**
✅ **Your video:** Mostly slow motion, but can still emphasize the fast parts
✅ **Method:** Custom weights - low for slow bands, high for fast bands
✅ **Advanced:** Combine with spatial masking for ultimate control

Try the example script above to see it in action! 🎯

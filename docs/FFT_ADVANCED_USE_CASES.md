# Advanced FFT Use Cases - Selective Motion Control

## Your Question: Fast Hands vs. Slow Body

**Scenario:** Human dancing where:
- Body doesn't move much (slow/static)
- Hands move very fast

**Can FFT control these differently?** → **YES, absolutely!**

## How FFT Separates Motion

### Temporal Frequency Decomposition

FFT separates motion by **speed (temporal frequency)**, not location:

| Motion Type | Speed | Frequency | FFT Band |
|-------------|-------|-----------|----------|
| Body sway/drift | Slow | 0.1-1 Hz | Band 0-1 (Low) |
| Walking, dance steps | Medium | 1-5 Hz | Band 1-2 (Mid) |
| Hand gestures | Fast | 5-15 Hz | Band 2-3 (High) |
| Finger movements | Very fast | 10-15 Hz | Band 3 (Very High) |

**Key insight:** Fast hand movements → High frequency bands, Slow body → Low frequency bands

## Use Case 1: Amplify Hand Motion, Stabilize Body

### Scenario
Dancing video where:
- Body barely moves (you want to remove/minimize)
- Hands move fast (you want to emphasize)

### Solution
```bash
# Custom weights: Remove low freq (body), keep high freq (hands)
python frequency_motion_transfer.py dance.mp4 \
    --custom_weights 0.0,0.3,1.0,1.5 \
    --output_dir hands_emphasized \
    --fps 30
```

**What this does:**
- Band 0 (0.1-0.68 Hz): Weight 0.0 → Remove body drift
- Band 1 (0.68-4.6 Hz): Weight 0.3 → Minimize body sway
- Band 2 (4.6-10 Hz): Weight 1.0 → Keep hand gestures
- Band 3 (10-15 Hz): Weight 1.5 → **AMPLIFY** fast hand movements!

**Result:** Body appears more static, hand movements exaggerated

## Use Case 2: Keep Body, Remove Hand Jitter

### Scenario
Performance video where:
- Body movements are good (choreography)
- Hands have nervous jitter/shake

### Solution
```bash
# Keep low-mid freq (body), remove high freq (jitter)
python frequency_motion_transfer.py performance.mp4 \
    --custom_weights 1.0,1.0,0.5,0.0 \
    --output_dir smooth_hands \
    --fps 30
```

**Result:** Smooth body motion, jittery hands smoothed out

## Use Case 3: Spatial + Frequency Combination (ADVANCED)

### The Ultimate Control
What if you want **different frequency filtering for different body parts?**

**Example:** Keep hand high-frequency motion, remove body high-frequency shake

### Solution: Spatial Masking + FFT

```python
import numpy as np
import cv2
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

# 1. Load your flow
flow = np.load('dance_flows.npy')  # Shape: (T, 2, H, W)

# 2. Create spatial mask (hands vs body)
H, W = flow.shape[2], flow.shape[3]
mask = np.zeros((H, W))

# Define hand regions (example coordinates)
# Top portion = hands, bottom = body
mask[:H//3, :] = 1.0  # Hands region

# 3. Apply FFT decomposition
editor = FrequencyMotionEditor(flow, fps=30)
editor.decompose_frequencies(num_bands=4)

# 4. Different frequency weights for different regions
hand_weights = [0.2, 0.5, 1.0, 1.2]  # Emphasize fast motion
body_weights = [1.0, 0.8, 0.3, 0.0]  # Keep slow, remove fast

# 5. Apply spatial frequency editing
filtered_flow = editor.apply_spatial_frequency_edit(
    mask=mask,
    band_weights_fg=hand_weights,  # Hands
    band_weights_bg=body_weights   # Body
)

# 6. Save
np.save('spatially_filtered_flow.npy', filtered_flow)
```

**What this does:**
- Hands region: Keeps/amplifies high frequency (fast gestures)
- Body region: Removes high frequency (stabilizes shake)

## Real-World Examples

### Example 1: Sign Language Video
```python
# Emphasize hand movements (high freq), minimize body sway (low freq)
weights = [0.1, 0.3, 1.0, 1.5]
```
**Before:** Body sways, hands gesture
**After:** Body stable, hand gestures emphasized

### Example 2: Conductor
```python
# Keep baton movements (high freq), remove body drift (low freq)
weights = [0.0, 0.2, 1.0, 1.2]
```
**Before:** Body drifts, baton moves
**After:** Body static, baton motion clear

### Example 3: Drummer
```python
# Keep stick movements (very high freq), stabilize body (low freq)
weights = [0.3, 0.5, 1.0, 1.5]
```
**Before:** Body sways with rhythm, sticks blur
**After:** Body stable, stick motion sharp

## Understanding Weights

### Weight Values
```python
weights = [band0, band1, band2, band3]
```

- `0.0` = Remove completely
- `0.5` = Reduce by half
- `1.0` = Keep as is (neutral)
- `1.5` = Amplify by 50%
- `2.0` = Double the motion

### Weight Patterns

**Remove Body, Keep Hands:**
```python
[0.0, 0.3, 1.0, 1.5]  # Progressive increase
```

**Remove Shake, Keep Everything Else:**
```python
[1.0, 1.0, 0.8, 0.0]  # Remove only highest band
```

**Emphasize Fast Motion:**
```python
[0.5, 0.7, 1.2, 2.0]  # Amplify high frequencies
```

**Smooth Everything:**
```python
[1.0, 0.8, 0.3, 0.0]  # Keep only low frequencies
```

## Limitations and When It Won't Work

### ❌ When FFT Alone Doesn't Help

**Case 1: Same Speed, Different Location**
- Left hand and right hand both move at same speed
- Just in different places
- FFT can't distinguish (need spatial masking)

**Case 2: Continuous Slow Motion**
- Everything moves slowly
- No frequency separation
- FFT won't help much

**Case 3: Chaotic Random Motion**
- No clear frequency structure
- FFT less effective

### ✅ When to Use Spatial + Frequency

If you want to control:
1. Different body parts differently
2. Foreground vs background
3. Left vs right side

Use the spatial masking approach shown above!

## Practical Workflow

### Step 1: Analyze Your Video

```bash
python frequency_motion_transfer.py your_video.mp4 \
    --output_dir analysis \
    --analyze_only \
    --fps 30
```

Check `analysis/motion_analysis.json` to see:
- Which bands have hand motion (look for high variance in high bands)
- Which bands have body motion (look for variance in low bands)

### Step 2: Design Your Weights

Based on analysis:
```python
# If analysis shows:
# Band 0: Low magnitude (body drift) → Remove (0.0)
# Band 1: Medium magnitude (body sway) → Reduce (0.3)
# Band 2: High magnitude (hand gestures) → Keep (1.0)
# Band 3: Very high magnitude (fast hands) → Amplify (1.5)

weights = [0.0, 0.3, 1.0, 1.5]
```

### Step 3: Apply Filtering

```bash
python frequency_motion_transfer.py your_video.mp4 \
    --custom_weights 0.0,0.3,1.0,1.5 \
    --output_dir custom_filtered \
    --fps 30
```

### Step 4: Check Results

```python
import numpy as np

original = np.load('analysis/raw_flows.npy')
filtered = np.load('custom_filtered/clean_flow.npy')

# Compare per-band magnitude
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

editor_orig = FrequencyMotionEditor(original, fps=30)
bands_orig = editor_orig.decompose_frequencies(num_bands=4)

editor_filt = FrequencyMotionEditor(filtered, fps=30)
bands_filt = editor_filt.decompose_frequencies(num_bands=4)

for i in range(4):
    mag_orig = np.sqrt(bands_orig[i][:,0]**2 + bands_orig[i][:,1]**2).mean()
    mag_filt = np.sqrt(bands_filt[i][:,0]**2 + bands_filt[i][:,1]**2).mean()
    print(f"Band {i}: {mag_orig:.2f} → {mag_filt:.2f}")
```

## Your Specific Case

**Your video:** Fast hands, slow body

### Quick Test

```bash
# Test 1: Analyze
python frequency_motion_transfer.py your_video.mp4 \
    --output_dir test_analysis \
    --analyze_only

# Check which bands have the hand motion
cat test_analysis/motion_analysis.json

# Test 2: Remove body, emphasize hands
python frequency_motion_transfer.py your_video.mp4 \
    --custom_weights 0.0,0.2,1.0,1.5 \
    --output_dir hands_emphasized

# Test 3: Smooth hands, keep body
python frequency_motion_transfer.py your_video.mp4 \
    --custom_weights 1.0,1.0,0.5,0.0 \
    --output_dir hands_smoothed
```

## Advanced: Temporal + Spatial Control

For ultimate control (different body parts, different frequencies):

```python
# Create hand detection mask (you can use segmentation, or manual)
import numpy as np

# Example: Simple spatial mask
H, W = 480, 720
mask = np.zeros((H, W))

# Hands are in top 1/3 of frame
mask[:H//3, :] = 1.0

# Apply
from CommonSource.frequency_motion_editor import FrequencyMotionEditor

flow = np.load('your_flows.npy')
editor = FrequencyMotionEditor(flow, fps=30)
editor.decompose_frequencies(num_bands=4)

# Hands: Keep fast motion
# Body: Remove fast motion
result = editor.apply_spatial_frequency_edit(
    mask=mask,
    band_weights_fg=[0.3, 0.5, 1.0, 1.5],  # Hands
    band_weights_bg=[1.0, 0.8, 0.3, 0.0]   # Body
)

np.save('spatially_controlled_flow.npy', result)
```

## Summary

### Yes, FFT Can Control Fast Hands vs Slow Body!

**How it works:**
- Fast hand movements → High frequency bands (2-3)
- Slow body movements → Low frequency bands (0-1)
- FFT separates them by temporal frequency

**What you can do:**
- Amplify hand motion: High weights on high bands
- Stabilize body: Low weights on low bands
- Mix and match: Custom weights per band
- Ultimate control: Spatial masks + frequency weights

**Your case (fast hands, slow body):**
```bash
# Remove body drift, emphasize hand gestures
--custom_weights 0.0,0.3,1.0,1.5
```

This is exactly what FFT was designed for! 🎯

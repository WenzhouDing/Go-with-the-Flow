# src/core/frequency/ - FFT Motion Editing

Core modules for frequency-based motion decomposition and editing.

## Overview

This package implements **temporal frequency decomposition** of optical flow using 1D FFT. It allows separating and controlling motion at different speeds independently.

**Key Capability:** Control fast hand gestures vs slow body motion separately!

## Modules

### frequency_motion_editor.py

Main class for FFT-based motion editing.

**Class: FrequencyMotionEditor**

```python
from src.core.frequency.frequency_motion_editor import FrequencyMotionEditor
import numpy as np

# Initialize with optical flow
flow = np.load('flow.npy')  # Shape: (T, 2, H, W)
editor = FrequencyMotionEditor(flow, fps=12)

# Decompose into frequency bands
bands = editor.decompose_frequencies(num_bands=4)

# Apply custom weights to each band
weights = [0.0, 0.3, 1.5, 2.0]  # [slow, medium, fast, very_fast]
filtered_flow = editor.reconstruct_selective(weights)

# Save result
np.save('filtered_flow.npy', filtered_flow)
```

**Methods:**

- `decompose_frequencies(num_bands=4)` - Decompose flow into frequency bands using 1D temporal FFT
- `reconstruct_selective(weights)` - Reconstruct flow with weighted frequency bands
- `get_band_info()` - Get frequency ranges and statistics for each band
- `visualize_spectrum()` - Plot frequency spectrum analysis

**Frequency Bands (12fps video):**
- Band 0: 0.1-0.68 Hz (slow drift, body sway)
- Band 1: 0.68-3.1 Hz (medium motion, walking)
- Band 2: 3.1-5.6 Hz (fast gestures, hand movements)
- Band 3: 5.6-6.0 Hz (very fast motion, shake)

### motion_classifier.py

Automatic motion classification and preset recommendation.

**Class: MotionClassifier**

```python
from src.core.frequency.motion_classifier import MotionClassifier

# Analyze motion in optical flow
classifier = MotionClassifier(flow, fps=12)
analysis = classifier.classify_motion()

print(f"Motion type: {analysis['primary_type']}")
print(f"Recommended preset: {analysis['recommended_preset']}")
print(f"Confidence: {analysis['confidence']:.2f}")
```

**Motion Classifications:**
- **Camera Shake:** High energy in band 3 (>6 Hz)
- **Body Movement:** Dominant energy in bands 0-1 (<3 Hz)
- **Micro Motion:** Low overall magnitude, distributed energy

**Recommended Presets:**
- `remove_shake` - If camera shake detected
- `hands_emphasized` - If body movement detected
- `ultra_smooth` - If micro motion detected

## Usage Examples

### Example 1: Emphasize Hand Gestures

Remove slow body motion, amplify fast hand movements:

```python
from src.core.frequency.frequency_motion_editor import FrequencyMotionEditor
import numpy as np

flow = np.load('results/warped_noise/dance/flows_dxdy.npy')
editor = FrequencyMotionEditor(flow, fps=12)
editor.decompose_frequencies(num_bands=4)

# Weights: [slow, medium, fast, very_fast]
weights = [0.0, 0.3, 1.5, 2.0]  # Remove slow, amplify fast
filtered = editor.reconstruct_selective(weights)

np.save('hands_emphasized_flow.npy', filtered)
```

**Result:** Body appears more static, hand movements more prominent

### Example 2: Ultra Smooth Motion

Keep only slow body motion, remove all fast movements:

```python
weights = [1.0, 0.8, 0.0, 0.0]  # Keep slow, remove fast
filtered = editor.reconstruct_selective(weights)
```

**Result:** Smooth, fluid motion without jitter

### Example 3: Extreme Hand Emphasis

Show ONLY fast hand movements:

```python
weights = [0.0, 0.0, 2.0, 3.0]  # Remove all slow, double fast
filtered = editor.reconstruct_selective(weights)
```

**Result:** Dramatic hand gestures, body disappears

### Example 4: Remove Camera Shake

Stabilize shaky footage:

```python
weights = [1.0, 1.0, 0.8, 0.0]  # Keep slow/medium, remove shake
filtered = editor.reconstruct_selective(weights)
```

**Result:** Stable footage without high-frequency jitter

## Technical Details

### FFT Implementation

**1D Temporal FFT:**
- Applied independently to each pixel's motion over time
- Flow shape: (T, 2, H, W) → FFT along T dimension
- Separates into frequency components

**Band Separation:**
- Logarithmic spacing for perceptually meaningful bands
- Each band represents different motion speeds
- Nyquist frequency = fps / 2

**Reconstruction:**
- Weighted sum of frequency bands
- Inverse FFT to return to spatial domain
- Preserves spatial structure while controlling temporal frequencies

### Performance

**Memory Usage:**
- Stores original flow + decomposed bands
- ~5x memory overhead during processing
- Recommended: Process at reduced resolution (240x360)

**Speed:**
- FFT decomposition: ~2 seconds for 49 frames @ 240x360
- Reconstruction: < 1 second
- Total overhead: ~3 seconds per video

## Integration with Pipeline

### Standard Integration

```bash
# 1. Create warped noise (gets flow at 240x360)
python src/pipeline/make_warped_noise.py \
    data/videos/my_video.mp4 \
    results/warped_noise/my_video/

# 2. Apply FFT filtering directly
python -c "
from src.core.frequency.frequency_motion_editor import FrequencyMotionEditor
import numpy as np

flow = np.load('results/warped_noise/my_video/flows_dxdy.npy')
editor = FrequencyMotionEditor(flow, fps=12)
editor.decompose_frequencies(num_bands=4)

weights = [0.0, 0.3, 1.5, 2.0]  # hands_emphasized
filtered = editor.reconstruct_selective(weights)

np.save('results/warped_noise/my_video/flows_dxdy_original.npy', flow)
np.save('results/warped_noise/my_video/flows_dxdy.npy', filtered)
"

# 3. Generate video
python src/pipeline/cut_and_drag_inference.py \
    results/warped_noise/my_video/ \
    results/generated/my_video/output.mp4
```

### Using Utility Tools

```bash
# Extract and analyze with FFT (high resolution)
python src/pipeline/process_dynamite_video.py

# Downsample to match warped noise resolution
python src/tools/resize_filtered_flow.py \
    results/fft_analysis/dynamite/hands_emphasized_flow.npy \
    240 360

# Integrate into warped noise
python src/tools/integrate_filtered_flow.py integrate \
    results/fft_analysis/dynamite/hands_emphasized_flow_240x360.npy \
    results/warped_noise/dynamite/
```

## Common Presets

### Preset: hands_emphasized
```python
weights = [0.0, 0.3, 1.5, 2.0]
```
- Remove slow drift (band 0)
- Reduce medium motion (band 1 × 0.3)
- Amplify fast gestures (band 2 × 1.5)
- Double very fast motion (band 3 × 2.0)
- **Use case:** Dancing, sign language, energetic gestures

### Preset: ultra_smooth
```python
weights = [1.0, 0.8, 0.0, 0.0]
```
- Keep slow motion (band 0)
- Slight reduction of medium (band 1 × 0.8)
- Remove all fast motion (bands 2-3 × 0.0)
- **Use case:** Smooth, flowing movements

### Preset: extreme_hands
```python
weights = [0.0, 0.0, 2.0, 3.0]
```
- Remove ALL slow motion (bands 0-1 × 0.0)
- Show ONLY fast gestures (bands 2-3 boosted)
- **Use case:** Dramatic effect, emphasize action

### Preset: remove_shake
```python
weights = [1.0, 1.0, 0.8, 0.0]
```
- Keep slow and medium motion (bands 0-1)
- Reduce fast motion (band 2 × 0.8)
- Remove shake (band 3 × 0.0)
- **Use case:** Stabilizing shaky camera

## Dependencies

```python
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
```

All dependencies included in `flow_warp` conda environment.

## Testing

```bash
conda activate flow_warp
pytest tests/test_frequency_motion.py -v
```

**Test Coverage:**
- FFT decomposition correctness
- Frequency band separation
- Reconstruction accuracy
- Motion classification
- Preset application

## References

- **1D Temporal FFT:** Separates motion by speed
- **Logarithmic Band Spacing:** Perceptually meaningful divisions
- **HSV Flow Visualization:** Hue=direction, Brightness=speed

## Future Enhancements

- [ ] Spatial masking for region-specific filtering
- [ ] Adaptive band selection based on video content
- [ ] Real-time preview of filtered motion
- [ ] GUI for weight adjustment
- [ ] Pre-trained motion presets for common scenarios

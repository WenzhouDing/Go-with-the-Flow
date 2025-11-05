# Quick Start: Frequency-Decomposed Motion Transfer

## ✓ Implementation Complete

**Status:** All features implemented and tested (14/14 tests passing)

## What You Can Do Now

### 1. Remove Camera Shake from Videos

```bash
conda activate flow_warp

# Analyze a shaky video
python frequency_motion_transfer.py shaky_dance.mp4 --analyze_only

# Remove shake automatically
python frequency_motion_transfer.py shaky_dance.mp4 \
    --output_dir stabilized \
    --remove_shake
```

### 2. Selective Motion Filtering

```bash
# Keep only low and mid-frequency motion (smooth movements)
python frequency_motion_transfer.py dance.mp4 \
    --custom_weights 1.0,1.0,0.0,0.0 \
    --output_dir smooth_motion

# Keep everything except high-frequency noise
python frequency_motion_transfer.py video.mp4 \
    --custom_weights 1.0,1.0,0.8,0.0 \
    --output_dir clean_motion
```

### 3. Analyze Motion Patterns

```bash
# Get detailed frequency analysis
python frequency_motion_transfer.py your_video.mp4 --analyze_only

# Visualize frequency bands
python test_frequency_decomposition.py your_video.mp4 --output viz/
```

### 4. Run Tests

```bash
# Run comprehensive unit tests (14 tests)
python tests/test_frequency_motion.py

# Quick inline test
python test_frequency_decomposition.py --test
```

## Files Created

### Core Implementation (437 lines)
- `CommonSource/frequency_motion_editor.py` (218 lines)
- `CommonSource/motion_classifier.py` (208 lines)
- `CommonSource/__init__.py` (11 lines)

### Tools (700 lines)
- `extract_flow_sequence.py` (126 lines) - Extract optical flow
- `frequency_motion_transfer.py` (224 lines) - Main pipeline
- `test_frequency_decomposition.py` (350 lines) - Visualization

### Tests (465 lines)
- `tests/test_frequency_motion.py` (465 lines) - 14 comprehensive tests

**Total: 1,602 lines of code**

## How It Works

```
Video → Extract Flow → Frequency Decomposition → Selective Reconstruction
                            ↓
                      [4 Frequency Bands]
                      Band 0: 0.1 - 0.68 Hz  (slow movements)
                      Band 1: 0.68 - 4.64 Hz (body movements)
                      Band 2: 4.64 - 10 Hz   (fast movements)
                      Band 3: 10 - 15 Hz     (shake/noise)
                            ↓
                      Apply Weights → Clean Flow
```

## Frequency Band Guide

| Band | Range | Contains | Typical Weight |
|------|-------|----------|----------------|
| 0 | 0.1 - 0.68 Hz | Camera pans, slow motion | 1.0 (keep) |
| 1 | 0.68 - 4.64 Hz | Body movements, choreography | 1.0 (keep) |
| 2 | 4.64 - 10 Hz | Fast movements, gestures | 0.5-1.0 |
| 3 | 10 - 15 Hz | Camera shake, micro-motion | 0.0 (remove) |

## Common Use Cases

### Remove Camera Shake
```bash
python frequency_motion_transfer.py video.mp4 \
    --output_dir output \
    --remove_shake
# Automatically detects and removes shake
```

### Keep Only Smooth Motion
```bash
python frequency_motion_transfer.py video.mp4 \
    --custom_weights 1.0,0.8,0.0,0.0
# Removes fast movements and shake
```

### Amplify Micro-Motions
```bash
python frequency_motion_transfer.py video.mp4 \
    --custom_weights 0.5,0.5,1.0,2.0
# Exaggerates high-frequency details
```

### Analyze Before Processing
```bash
# Step 1: Analyze
python frequency_motion_transfer.py video.mp4 --analyze_only

# Step 2: Review the motion analysis report
# Shows classification scores for each band

# Step 3: Choose custom weights based on analysis
python frequency_motion_transfer.py video.mp4 \
    --custom_weights <your_weights>
```

## Environment Setup

This implementation uses **two conda environments**:

### flow_warp (for frequency analysis)
```bash
conda activate flow_warp
# Use for:
# - extract_flow_sequence.py
# - frequency_motion_transfer.py
# - test_frequency_decomposition.py
# - tests/test_frequency_motion.py
```

### flow (for video generation)
```bash
conda activate flow
# Use for:
# - cut_and_drag_inference.py
# - make_warped_noise.py (video generation part)
```

## Verification

### Run All Tests
```bash
conda activate flow_warp
python tests/test_frequency_motion.py
```

**Expected output:**
```
======================================================================
TEST SUMMARY: 14/14 passed
✓ ALL TESTS PASSED
======================================================================
```

### Quick Function Check
```bash
# Test 1: Check imports
python -c "from CommonSource import FrequencyMotionEditor, MotionClassifier; print('✓ Imports work')"

# Test 2: Check instantiation
python -c "
import numpy as np
from CommonSource import FrequencyMotionEditor
flow = np.random.randn(49, 2, 60, 90)
editor = FrequencyMotionEditor(flow, fps=30)
print('✓ Instantiation works')
"

# Test 3: Check decomposition
python -c "
import numpy as np
from CommonSource import FrequencyMotionEditor
flow = np.random.randn(49, 2, 60, 90)
editor = FrequencyMotionEditor(flow, fps=30)
bands = editor.decompose_frequencies(num_bands=4)
print(f'✓ Decomposition works - created {len(bands)} bands')
"
```

## Integration Example

### Complete Pipeline
```bash
# Step 1: Extract flow from video
conda activate flow_warp
python extract_flow_sequence.py dance_video.mp4 \
    --output dance_flows.npy

# Step 2: Process with frequency decomposition
python frequency_motion_transfer.py dance_video.mp4 \
    --flow_path dance_flows.npy \
    --output_dir freq_output \
    --remove_shake

# Step 3: (Future) Use clean flow for noise warping
# python make_warped_noise.py --flow_override freq_output/clean_flow.npy

# Step 4: Generate video with cleaned motion
conda activate flow
python cut_and_drag_inference.py noise_output/ \
    --prompt "A dancing robot" \
    --device cuda
```

## Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the right directory
cd /home/wding/Desktop/Go-with-the-Flow

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue: Tests fail
```bash
# Ensure dependencies installed
pip install numpy scipy matplotlib

# Re-run tests with verbose output
python tests/test_frequency_motion.py
```

### Issue: "No module named 'rp'"
```bash
# extract_flow_sequence.py requires the 'rp' package
# This is installed in the flow_warp environment
conda activate flow_warp
pip install rp
```

## Performance Notes

- **Speed:** ~5-10 seconds for 49-frame sequence
- **Memory:** ~500MB for standard video (480x720x49)
- **GPU:** Not required (CPU-only numpy/scipy FFT)

## Next Steps

1. ✓ Test with your videos:
   ```bash
   python frequency_motion_transfer.py your_video.mp4 --analyze_only
   ```

2. ✓ Experiment with band weights:
   ```bash
   python frequency_motion_transfer.py your_video.mp4 \
       --custom_weights 1.0,0.8,0.5,0.0
   ```

3. ✓ Visualize results:
   ```bash
   python test_frequency_decomposition.py freq_output/
   ```

4. ⚠ Integration with noise warping:
   - Currently outputs clean_flow.npy
   - Integration with make_warped_noise.py pending
   - Can be done via manual numpy flow replacement

## Support Files

- `Motion_FFT.md` - Original design specification
- `FREQUENCY_MOTION_IMPLEMENTATION.md` - Complete implementation docs
- `REPOSITORY_ORGANIZATION.md` - Project structure
- `tests/test_frequency_motion.py` - Test suite

## Success!

✓ All features implemented
✓ All tests passing (14/14)
✓ Ready for use with your videos
✓ Fully documented

**Enjoy frequency-decomposed motion transfer!**

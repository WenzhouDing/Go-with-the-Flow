# Archive

This directory contains previous implementations that have been archived for reference.

## Spatiotemporal Degradation

**Location:** `archive/spatiotemporal_degradation/`

**What it was:** An implementation that added per-pixel and per-frame control over motion strength in Go-with-the-Flow video generation. This allowed varying how much motion control was applied across space and time.

**Files:**
- `degradation_control.py` - Core implementation with DegradationConfig, DegradationProcessor classes
- `SPATIOTEMPORAL_DEGRADATION_README.md` - Full documentation
- `create_degradation_configs.py` - Example configuration generator
- `test_degradation_control.py` - Unit tests
- `configs/` - Pre-generated configuration files (20 examples)

**Why archived:** The project switched to a frequency-decomposed motion transfer approach (see `Motion_FFT.md`) which uses 3D FFT to decompose optical flow into frequency bands for selective motion filtering.

**Note:** Basic scalar degradation support remains in the main codebase (`cut_and_drag_inference.py`). The complex spatiotemporal patterns are what was archived.

**To use this implementation:** Copy the files back to the root directory and restore the imports in `cut_and_drag_inference.py`.

## Legacy Outputs

**Location:** `archive/legacy_outputs/`

Output directories from before repository reorganization (Nov 5, 2024). These were moved from the root directory to keep things organized.

**Directories:**
- `brown-bear/` - Test outputs for brown-bear video
- `comparison_outputs/` - Baseline comparison videos
- `fft_filtered/` - Early FFT filtering test data and analysis
- `my_warped_noise/` - Old warped noise folder
- `outputs/` - Generic old outputs

**Current alternatives:** Use `results/` with proper subdirectories:
- `results/generated/` for output videos
- `results/warped_noise/[video_name]/` for warped noise
- `results/fft_analysis/[video_name]/` for FFT results

## Test Data

**Location:** `archive/test_data/`

Large NPY arrays from FFT testing and development (206 MB total).

**Files:**
- `fft_filtered_flow.npy` (16 MB) - Early FFT filtered flow
- `test_extreme_hands.npy` (64 MB) - Test preset data
- `test_hands_emphasized.npy` (64 MB) - Test preset data
- `test_ultra_smooth.npy` (64 MB) - Test preset data

**Cleanup:** You can safely delete this directory to save disk space if not needed for comparison.

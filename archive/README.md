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

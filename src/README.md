# src/ - Source Code

All Python source code for the Go-with-the-Flow pipeline.

## Structure

```
src/
├── core/              # Core modules and libraries
│   └── frequency/    # FFT-based motion editing
│
├── pipeline/          # Main pipeline scripts
│   ├── make_warped_noise.py           # Create warped noise from video
│   ├── cut_and_drag_inference.py      # Video generation with CogVideoX
│   ├── cut_and_drag_gui.py            # GUI for inference
│   └── process_dynamite_video.py      # FFT analysis on dynamite
│
└── tools/             # Utility scripts
    ├── extract_flow_sequence.py       # Flow extraction tool
    ├── frequency_motion_transfer.py   # CLI for FFT filtering
    ├── resize_filtered_flow.py        # Flow resolution matching
    └── integrate_filtered_flow.py     # Replace flows in warped noise
```

## Module Overview

### Core Modules

**src/core/frequency/**
- FFT-based temporal frequency decomposition
- Motion classification and analysis
- See `src/core/frequency/README.md` for details

### Pipeline Scripts

**make_warped_noise.py**
```bash
python src/pipeline/make_warped_noise.py <video_path> <output_folder>
```
- Preprocesses video (resize, crop, 49 frames)
- Extracts optical flow using RAFT
- Warps Gaussian noise using flow
- Outputs: preprocessed video, flows, warped noise

**cut_and_drag_inference.py**
```bash
python src/pipeline/cut_and_drag_inference.py \
    <warped_noise_folder> \
    <output_video> \
    --degradation 0.0 \
    --num_inference_steps 50
```
- Loads warped noise and flows
- Runs CogVideoX diffusion model
- Generates motion-controlled video
- Supports text prompts and guidance

**process_dynamite_video.py**
```bash
python src/pipeline/process_dynamite_video.py
```
- Complete FFT pipeline for dynamite video
- Extracts flow, applies 4 presets
- Generates visualizations and comparisons
- Outputs to `dynamite_results/`

### Utility Tools

**extract_flow_sequence.py**
```bash
python src/tools/extract_flow_sequence.py <video_path> [output_path]
```
- Standalone optical flow extraction
- Uses RAFT model
- Saves flow arrays and visualizations

**frequency_motion_transfer.py**
```bash
python src/tools/frequency_motion_transfer.py \
    <input_flow.npy> \
    <output_flow.npy> \
    --weights 0.0 0.3 1.5 2.0
```
- CLI for applying FFT filtering
- Custom frequency band weights
- Batch processing support

**resize_filtered_flow.py**
```bash
python src/tools/resize_filtered_flow.py \
    <flow.npy> \
    <target_height> \
    <target_width>
```
- Downsample flow arrays
- Properly scales flow values
- Handles resolution mismatches

**integrate_filtered_flow.py**
```bash
python src/tools/integrate_filtered_flow.py integrate \
    <filtered_flow.npy> \
    <warped_noise_folder>
```
- Replaces flow in warped noise folder
- Backs up original flow
- Shows magnitude comparison

## Usage Patterns

### Standard Workflow

```bash
# 1. Create warped noise from video
python src/pipeline/make_warped_noise.py \
    data/videos/my_video.mp4 \
    results/warped_noise/my_video/

# 2. Generate video (no FFT)
python src/pipeline/cut_and_drag_inference.py \
    results/warped_noise/my_video/ \
    results/generated/my_video/output.mp4 \
    --degradation 0.0
```

### FFT Workflow

```bash
# 1. Extract and analyze flow with FFT
python src/pipeline/process_dynamite_video.py

# 2. Create warped noise
python src/pipeline/make_warped_noise.py \
    data/videos/samples/dynamite_clip.mp4 \
    results/warped_noise/dynamite/

# 3. Integrate FFT-filtered flow
python src/tools/integrate_filtered_flow.py integrate \
    results/fft_analysis/dynamite/hands_emphasized_flow_240x360.npy \
    results/warped_noise/dynamite/

# 4. Generate video
python src/pipeline/cut_and_drag_inference.py \
    results/warped_noise/dynamite/ \
    results/generated/dynamite/hands_emphasized.mp4 \
    --degradation 0.0
```

### Custom FFT Filtering

```python
from src.core.frequency.frequency_motion_editor import FrequencyMotionEditor
import numpy as np

# Load flow
flow = np.load('results/warped_noise/my_video/flows_dxdy.npy')

# Decompose into 4 bands
editor = FrequencyMotionEditor(flow, fps=12)
editor.decompose_frequencies(num_bands=4)

# Custom weights: [slow_drift, medium, fast_gestures, very_fast]
weights = [0.5, 1.0, 2.0, 0.5]
filtered = editor.reconstruct_selective(weights)

# Save
np.save('custom_filtered_flow.npy', filtered)
```

## Dependencies

### Conda Environments

**flow_warp** (optical flow & FFT):
```bash
conda activate flow_warp
# For: make_warped_noise.py, process_dynamite_video.py, tools/*
```

**flow** (video diffusion):
```bash
conda activate flow
# For: cut_and_drag_inference.py, cut_and_drag_gui.py
```

## Development

### Adding New Scripts

**Pipeline scripts** (`src/pipeline/`):
- Main workflows that users run directly
- Should have clear CLI interface
- Document in this README

**Tools** (`src/tools/`):
- Utility scripts for specific tasks
- Often used as part of larger workflows
- Should be modular and reusable

**Core modules** (`src/core/`):
- Reusable libraries and classes
- Imported by pipeline scripts and tools
- Should have comprehensive docstrings

### Import Conventions

```python
# For pipeline scripts and tools
from src.core.frequency.frequency_motion_editor import FrequencyMotionEditor
from src.core.frequency.motion_classifier import MotionClassifier

# For CommonSource modules (legacy)
import rp
rp.git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw
```

## Testing

Run unit tests:
```bash
conda activate flow_warp
pytest tests/test_frequency_motion.py -v
```

## Documentation

- **DYNAMITE_FFT_COMPLETE_GUIDE.md** - Complete workflow guide
- **QUICK_START_FFT.md** - Quick reference
- **REPOSITORY_STATUS.md** - Current implementation status
- **docs/** - Additional documentation

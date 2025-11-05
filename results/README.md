# results/ - Output Directory

This folder contains all pipeline outputs, organized by type.

## Structure

```
results/
├── flows/              # Extracted optical flows
│   └── [video_name]/  # Flow arrays and visualizations
│
├── fft_analysis/       # FFT analysis results
│   └── [video_name]/  # Frequency decomposition, visualizations, presets
│
├── warped_noise/       # Warped noise for diffusion
│   └── [video_name]/  # Noise arrays, flows, preview videos
│
└── generated/          # Final generated videos
    └── [video_name]/  # Output MP4 files
```

## Usage

### Flows Directory

Contains raw optical flow extracted from videos:

```bash
# Example output
results/flows/dynamite/
├── flows_dxdy.npy          # Raw flow (T, 2, H, W)
├── flow_visualization.mp4   # HSV-coded preview
└── flow_stats.json         # Magnitude and statistics
```

### FFT Analysis Directory

Contains frequency decomposition results:

```bash
# Example output
results/fft_analysis/dynamite/
├── original_flow.npy                    # Full resolution flow
├── original_flow.png                    # Visualization
├── motion_analysis.json                 # Frequency analysis
├── motion_spectrum.png                  # Analysis charts
│
├── hands_emphasized_flow.npy            # Filtered preset
├── hands_emphasized_flow_240x360.npy    # Downsampled version
├── hands_emphasized_comparison.png       # Before/after
│
├── ultra_smooth_flow.npy
├── extreme_hands_flow.npy
└── remove_shake_flow.npy
```

### Warped Noise Directory

Contains warped noise ready for diffusion:

```bash
# Example output
results/warped_noise/dynamite_hands_emphasized/
├── input.mp4                    # Preprocessed video
├── flows_dxdy.npy               # Flow (48, 2, 240, 360)
├── flows_dxdy_original.npy      # Backup if FFT applied
├── noises.npy                   # Warped noise (49, 60, 90, 16)
├── first_frame.png              # Preview
├── noise_video.mp4              # Noise visualization
└── visualization_video.mp4      # RGB + noise side-by-side
```

### Generated Directory

Contains final output videos:

```bash
# Example output
results/generated/dynamite/
├── hands_emphasized.mp4         # deg=0.0, 50 steps
├── hands_emphasized_prompt.mp4  # With text prompt
├── ultra_smooth.mp4
└── extreme_hands.mp4
```

## Notes

- This entire `results/` folder is gitignored
- Large binary files (*.npy, *.mp4) are not committed
- Organize by video name for clarity
- Keep FFT presets in separate warped noise folders

## Workflow Example

```bash
# 1. Extract flow
python src/tools/extract_flow_sequence.py \
    data/videos/my_video.mp4 \
    results/flows/my_video/

# 2. Apply FFT analysis
python src/pipeline/process_video_fft.py \
    results/flows/my_video/flows_dxdy.npy \
    results/fft_analysis/my_video/

# 3. Create warped noise
python src/pipeline/make_warped_noise.py \
    data/videos/my_video.mp4 \
    results/warped_noise/my_video/

# 4. Integrate FFT filter
python src/tools/integrate_filtered_flow.py integrate \
    results/fft_analysis/my_video/hands_emphasized_flow_240x360.npy \
    results/warped_noise/my_video/

# 5. Generate video
python src/pipeline/cut_and_drag_inference.py \
    results/warped_noise/my_video/ \
    results/generated/my_video/output.mp4 \
    --degradation 0.0
```

## Cleanup

To free up disk space:

```bash
# Remove intermediate flows (keep FFT analysis)
rm -rf results/flows/

# Remove warped noise after generation
rm -rf results/warped_noise/

# Keep only final generated videos
```

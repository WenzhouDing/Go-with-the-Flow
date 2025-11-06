# Segmentation Flow Scaling: Issue Analysis and Fix

## The Critical Bug

**Problem**: After applying segmentation-based flow scaling (foreground 1.5x, background 0.3x), the generated videos showed no noticeable difference from the original.

## Root Cause Analysis

###  Step 1: Understanding the Pipeline

The video generation pipeline has two distinct phases:

1. **Warped Noise Creation** (`make_warped_noise.py`):
   - Extracts optical flow from video
   - Generates random Gaussian noise
   - **Warps the noise using the flow** → creates `noises.npy`
   - Saves both `flows_dxdy.npy` (flow) and `noises.npy` (warped noise)

2. **Video Generation** (`cut_and_drag_inference.py`):
   - Loads `noises.npy` (line 194)
   - **Does NOT use `flows_dxdy.npy`**
   - Uses the pre-warped noise as latent init for diffusion

### Step 2: What Went Wrong

**Our incorrect approach:**
```bash
# Step 1: Scale the optical flow
python segment_and_scale_flow.py \
    --flow results/warped_noise/train/flows_dxdy.npy \
    --fg_scale 1.5 --bg_scale 0.3 \
    --output scaled_flow.npy

# Step 2: Copy old warped noise directory and replace flow file
cp -r results/warped_noise/train/ results/warped_noise/train_scaled/
cp scaled_flow.npy results/warped_noise/train_scaled/flows_dxdy.npy

# Step 3: Generate video
python cut_and_drag_inference.py \
    --warped_noise_dir results/warped_noise/train_scaled/
```

**Why this failed:**
- `noises.npy` was ALREADY WARPED with the ORIGINAL unscaled flow
- We only replaced `flows_dxdy.npy`, which isn't used during inference
- Inference loaded the OLD warped noise → no scaling effect!

### Step 3: Evidence

Check the timestamps:
```bash
$ ls -lh results/warped_noise/train_seg_scaled/
-rw-rw-r-- 1 wding wding 8.1M Nov  6 00:46 noises.npy      # OLD (warped with original flow)
-rw-rw-r-- 1 wding wding  32M Nov  6 01:09 flows_dxdy.npy  # NEW (scaled flow, but unused!)
```

The `noises.npy` was created BEFORE we scaled the flow, so it contains noise warped with the original unscaled flow.

## The Solution

We need to **re-warp the noise with the scaled flow** instead of reusing old warped noise.

### New Tool: `rewarp_noise_with_scaled_flow.py`

This tool:
1. Loads the scaled optical flow
2. Generates fresh Gaussian noise
3. Warps the noise using the SCALED flow
4. Saves the newly warped noise

### Correct Workflow

```bash
# Step 1: Extract optical flow from video
python src/pipeline/make_warped_noise.py \
    data/videos/train.mp4 \
    results/warped_noise/train_normal/

# Step 2: Scale flow based on segmentation
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/train_normal/flows_dxdy.npy \
    --mask data/videos_sam/train_sam2.mp4 \
    --fg_scale 1.5 \
    --bg_scale 0.3 \
    --output results/segmentation_scaled/train_fg1.5_bg0.3/scaled_flow.npy \
    --visualize results/visualizations/train_segmented_flow.mp4

# Step 3: RE-WARP noise with scaled flow (THE FIX!)
python src/tools/rewarp_noise_with_scaled_flow.py \
    --flow results/segmentation_scaled/train_fg1.5_bg0.3/scaled_flow.npy \
    --reference_dir results/warped_noise/train_normal/ \
    --output_dir results/warped_noise/train_seg_scaled_rewarp/ \
    --visualize

# Step 4: Generate video using re-warped noise
python src/pipeline/cut_and_drag_inference.py \
    --warped_noise_dir results/warped_noise/train_seg_scaled_rewarp/ \
    --output_path results/generated/train_scaled.mp4 \
    --prompt "a train moving through the countryside" \
    --seed 42
```

## Technical Details

### Why Noise Must Be Re-Warped

The diffusion model uses the warped noise as initialization for the latent space. The motion patterns are encoded IN the warped noise structure, not in the flow file. Therefore:

- **Correct**: Warp noise with scaled flow → scaled motion in latents → scaled motion in video
- **Wrong**: Use old warped noise + new flow file → old motion in latents → old motion in video

### NoiseWarper Class Usage

The `rewarp_noise_with_scaled_flow.py` tool uses the `NoiseWarper` class from `noise_warp.py`:

```python
# Initialize warper with random Gaussian noise
warper = nw.NoiseWarper(
    noise_C,      # Number of channels (16 for latents)
    noise_H,      # Height (60 for 480p latent)
    noise_W,      # Width (90 for 720p latent)
    device=device,
    scale_factor=1
)

# Get first frame (initial random noise)
torch_noises.append(warper.noise.cpu().permute(1, 2, 0))

# Warp through sequence
for t in range(1, T):
    dx = scaled_flow[t-1, 0]  # Horizontal flow
    dy = scaled_flow[t-1, 1]  # Vertical flow

    # Apply warping (modifies internal state)
    warper(dx, dy)

    # Get warped noise for this frame
    warped = warper.noise.cpu().permute(1, 2, 0)
    torch_noises.append(warped)

# Save warped noise
np.save(output_path, np.stack(torch_noises))
```

## Results

After fixing:
- **Scaled flow**: Mean magnitude 2.616 pixels/frame
- **Re-warped noise**: 48 frames, shape (48, 60, 90, 16)
- **Output directory**: `results/warped_noise/train_seg_scaled_rewarp/`

Now the generated videos WILL show the scaling effect:
- Train (foreground) motion amplified by 1.5x
- Background motion reduced to 0.3x

## Key Takeaway

**The motion is in the noise, not in the flow file!**

When modifying optical flow for motion control:
1. Scale/filter the flow
2. **Re-warp fresh noise with the modified flow**
3. Use the re-warped noise for generation

Simply replacing `flows_dxdy.npy` doesn't work because that file is only used during the initial warped noise creation, not during inference.

## Files

- **Analysis tool**: `src/tools/segment_and_scale_flow.py`
- **Fix tool**: `src/tools/rewarp_noise_with_scaled_flow.py`
- **Documentation**: `src/tools/SEGMENTATION_FLOW_SCALING_README.md`

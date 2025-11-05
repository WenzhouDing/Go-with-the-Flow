# Optical Flow Visualization Guide

## What is Optical Flow Visualization?

Optical flow is visualized using **HSV color space** where:
- **Hue (color)** = Direction of motion
- **Saturation** = Always 255 (full saturation)
- **Value (brightness)** = Magnitude of motion

This creates an intuitive visualization where:
- **Color** shows which direction pixels are moving
- **Brightness** shows how fast they're moving

## Color Wheel Legend

```
        Red (0°)
           ↑
           |
Cyan ←-----+----→ Yellow
(180°)     |     (60°)
           ↓
        Blue (120°)
```

**Direction mapping:**
- **Red** (0°) → Moving right
- **Yellow** (60°) → Moving down-right
- **Green** (120°) → Moving down
- **Cyan** (180°) → Moving left
- **Blue** (240°) → Moving up-left
- **Magenta** (300°) → Moving up

**Brightness mapping:**
- **Dark** = Little to no motion
- **Bright** = Strong motion
- **Very bright** = Very fast motion

## Interpreting the Visualizations

### Original Flow
Shows the raw optical flow extracted from the video:
- **Uniform colors across regions** = Camera motion (everything moves together)
- **Different colors in different regions** = Object motion (things move independently)
- **High brightness** = Fast motion (hand gestures, camera shake)
- **Low brightness** = Slow motion (body sway, drift)

### FFT-Filtered Flow

Different presets emphasize different aspects:

#### 1. Hands Emphasized (`[0.0, 0.3, 1.5, 2.0]`)
- **Brighter high-frequency regions** = Fast motions amplified
- **Darker low-frequency regions** = Slow drift removed
- **Sharp color transitions** = Quick movements preserved
- Use case: Emphasize hand gestures, remove body sway

#### 2. Ultra Smooth (`[1.0, 0.8, 0.0, 0.0]`)
- **Smooth color gradients** = Only slow motion kept
- **Less sharp transitions** = Fast movements removed
- **Lower overall brightness** = Reduced motion energy
- Use case: Stabilize shaky footage, smooth dance movements

#### 3. Extreme Hands (`[0.0, 0.0, 2.0, 3.0]`)
- **Only bright regions remain** = Only fast motion
- **Dark everywhere else** = Body motion completely removed
- **Very localized bright spots** = Isolated fast movements
- Use case: Isolate hand/finger movements, remove all body motion

#### 4. Remove Shake (`[1.0, 1.0, 0.8, 0.0]`)
- **Similar to original but smoother** = Most motion preserved
- **Less high-frequency noise** = Jitter/shake removed
- **Natural-looking** = Subtle improvement
- Use case: Remove camera shake while keeping content motion

## Side-by-Side Comparison

The comparison images show:
- **Top row** = Original flow (before FFT)
- **Bottom row** = Filtered flow (after FFT)

**What to look for:**
1. **Magnitude changes** = Numbers below each frame show average motion
2. **Color preservation** = Check if motion directions are preserved
3. **Brightness changes** = See which frequency components were enhanced/removed
4. **Spatial patterns** = Notice which regions are affected more

## Example Analysis

**Dance video with fast hands:**

**Original Flow:**
```
Frame 0:  Mag: 2.35  (Medium brightness, mixed colors)
Frame 12: Mag: 4.82  (High brightness, hand region very bright)
Frame 24: Mag: 1.56  (Low brightness, slow body sway)
```

**After Hands Emphasized [0.0, 0.3, 1.5, 2.0]:**
```
Frame 0:  Mag: 0.85  (Reduced, body drift removed)
Frame 12: Mag: 6.12  (Increased! Hand motion amplified)
Frame 24: Mag: 0.42  (Much reduced, slow motion filtered out)
```

**Interpretation:**
- Frame 12 (hands moving fast): Magnitude increased from 4.82 → 6.12
- Frame 24 (body sway): Magnitude decreased from 1.56 → 0.42
- FFT successfully separated and controlled fast vs slow motion!

## Motion Spectrum Plot

The 4-panel spectrum visualization shows:

### Panel 1: Magnitude per Band (Bar chart)
- **X-axis:** Frequency bands (0=slowest, 3=fastest)
- **Y-axis:** Mean magnitude of motion in that band
- **Interpretation:**
  - High bar in band 0-1 = Mostly slow motion (body sway, drift)
  - High bar in band 2-3 = Lots of fast motion (gestures, shake)

### Panel 2: Temporal Variance (Bar chart)
- Shows how much each frequency band varies over time
- **High variance** = Motion is intermittent (comes and goes)
- **Low variance** = Motion is constant throughout video

### Panel 3: Frequency Ranges (Labeled bar chart)
- Same as Panel 1 but with frequency labels (Hz)
- Shows the actual Hz range each band covers
- Example: "0.1-0.68 Hz" (very slow), "5.6-6.0 Hz" (very fast)

### Panel 4: Motion Over Time (Line plot)
- Shows how total motion magnitude changes across frames
- **Peaks** = Moments of high activity
- **Valleys** = Moments of low activity
- **Smooth curve** = Consistent motion
- **Spiky curve** = Intermittent motion

## Reading the Analysis Report

When the script runs, it prints:

```
Motion Analysis Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Band 0 (0.1-0.68 Hz):
  Mean magnitude: 0.58
  Temporal variance: 0.0234
  Spatial variance: 45.23
  → Interpretation: Slow body sway
```

**What each metric means:**

1. **Mean magnitude** = Average motion strength in this frequency range
   - < 0.5: Weak motion
   - 0.5-2.0: Moderate motion
   - > 2.0: Strong motion

2. **Temporal variance** = How much this band's motion varies over time
   - < 0.01: Very consistent
   - 0.01-0.1: Somewhat variable
   - > 0.1: Highly intermittent

3. **Spatial variance** = How uniform the motion is across the frame
   - Low: Motion is the same everywhere (camera shake)
   - High: Motion varies by location (object motion)

## Output Files Explained

After running `process_dynamite_video.py`, you get:

### Flow Files (`.npy`)
- `original_flow.npy` - Raw optical flow (48, 2, 480, 720)
- `*_flow.npy` - Filtered flows for each preset
- Load with: `flow = np.load('original_flow.npy')`

### Visualization Images (`.png`)
- `original_flow.png` - 5 frames of original flow
- `*_flow.png` - 5 frames of each filtered version
- `*_comparison.png` - Side-by-side before/after
- `motion_spectrum.png` - 4-panel frequency analysis

### Videos (`.mp4`)
- `original_flow.mp4` - Full video of original flow
- `*_flow.mp4` - Full video of filtered flow
- Play these to see temporal dynamics

### Analysis (`.json`)
- `motion_analysis.json` - Numerical analysis results
- Contains per-band statistics and frequency ranges
- Load with: `analysis = json.load(open('motion_analysis.json'))`

## Practical Use Cases

### Use Case 1: Dance Performance
**Goal:** Keep dance moves, remove camera shake

**Look for:**
- Original: High magnitude in band 3 (shake)
- Filtered (remove_shake): Band 3 reduced, bands 0-2 preserved
- Visual: Smoother flow video, same colors but less noise

### Use Case 2: Sign Language
**Goal:** Emphasize hand signs, stabilize body

**Look for:**
- Original: Bands 0-1 dominant (body sway)
- Filtered (hands_emphasized): Band 2-3 amplified, band 0 removed
- Visual: Hand regions brighter, body regions darker

### Use Case 3: Action Sports
**Goal:** Remove jittery camera, keep subject motion

**Look for:**
- Original: Uniform high-frequency across frame (camera shake)
- Filtered (remove_shake): Lower spatial variance in band 3
- Visual: Subject motion preserved, background more stable

## Tips for Choosing Weights

Based on the spectrum plot:

1. **Identify dominant bands**
   - Check which bands have highest magnitude in bar chart

2. **Design weights accordingly**
   - Want to keep that motion? Use weight ≥ 1.0
   - Want to remove that motion? Use weight ≈ 0.0
   - Want to amplify? Use weight > 1.0

3. **Iterate and refine**
   - Start with preset (e.g., `[0.0, 0.3, 1.5, 2.0]`)
   - Check comparison images
   - Adjust weights based on visual results

## Common Patterns

**Remove camera shake (horizontal/vertical):**
```python
[1.0, 1.0, 0.5, 0.0]  # Keep low-mid, reduce high
```

**Emphasize fast hands:**
```python
[0.0, 0.3, 1.5, 2.0]  # Remove low, amplify high
```

**Ultra smooth:**
```python
[1.0, 0.8, 0.0, 0.0]  # Only keep lowest frequencies
```

**Isolate very fast motion:**
```python
[0.0, 0.0, 1.0, 2.0]  # Only bands 2-3
```

**Natural stabilization:**
```python
[0.8, 1.0, 1.0, 0.5]  # Slight reduction at extremes
```

## Understanding the Math

The visualization converts flow `(u, v)` to HSV:

```python
# Flow components
u = horizontal displacement
v = vertical displacement

# Convert to polar
magnitude = sqrt(u² + v²)
angle = atan2(v, u)

# Map to HSV
hue = angle * (180 / π) / 2      # Direction → Color
saturation = 255                  # Full saturation
value = normalize(magnitude)      # Magnitude → Brightness
```

This is why:
- Right motion (u>0, v=0) → Red (angle=0°)
- Down motion (u=0, v>0) → Green (angle=90°)
- Left motion (u<0, v=0) → Cyan (angle=180°)
- Up motion (u=0, v<0) → Magenta (angle=270°)

## Next Steps

After analyzing the visualizations:

1. **Choose best preset** based on your use case
2. **Integrate filtered flow** into generation pipeline
3. **Generate video** with filtered motion
4. **Compare results** visually

Example workflow:
```bash
# 1. Analyze
python process_dynamite_video.py

# 2. Integrate chosen preset
python integrate_filtered_flow.py integrate \
    dynamite_results/hands_emphasized_flow.npy \
    my_warped_noise/

# 3. Generate
conda activate flow
python cut_and_drag_inference.py \
    --folder_path my_warped_noise \
    --degradation 0.0 \
    --output_name dynamite_hands_emphasized.mp4
```

## Troubleshooting Visualizations

**Problem:** Flow looks mostly black
- **Cause:** Low motion magnitude
- **Fix:** Video might be too static, or needs higher contrast in visualization

**Problem:** Flow is all one color
- **Cause:** Camera motion (everything moves together)
- **Fix:** This is normal for camera shake - use remove_shake preset

**Problem:** Can't see difference between original and filtered
- **Cause:** Weights might be too conservative
- **Fix:** Try more extreme weights like `[0.0, 0.0, 2.0, 3.0]`

**Problem:** Filtered flow looks noisy
- **Cause:** Amplifying high frequencies amplifies noise
- **Fix:** Use smaller weights on high bands, or add smoothing

## Summary

**Optical flow visualization** is a powerful tool to:
- ✅ See what motion is in your video
- ✅ Understand frequency decomposition results
- ✅ Verify FFT filtering is working correctly
- ✅ Choose optimal weights for your use case

**Key insight:** Color = direction, brightness = speed. After FFT filtering, you should see brightness changes in specific frequency ranges while color patterns are preserved.

Happy visualizing! 🎨

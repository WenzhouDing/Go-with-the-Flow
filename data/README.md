# data/ - Input Data Directory

This folder contains all input data for the pipeline.

## Structure

```
data/
├── videos/
│   ├── samples/          # Sample/test videos
│   └── [your videos]     # Put your source videos here
│
└── test/                 # Test data for unit tests
```

## Usage

### Adding Input Videos

Put your raw video files in `data/videos/`:

```bash
# Example
cp /path/to/my_video.mp4 data/videos/
```

### Sample Videos

Test videos should go in `data/videos/samples/`:

```bash
# Example: dynamite_clip.mp4
data/videos/samples/
└── dynamite_clip.mp4
```

## Supported Formats

- **Video:** `.mp4`, `.avi`, `.mov`, `.webm`
- **Recommended:** H.264 MP4, 720p or higher
- **Duration:** 6-7 seconds for best results (see limitations in docs)

## Notes

- This entire `data/` folder is gitignored
- Do not commit videos to git (they're too large)
- Keep originals safe - pipeline creates copies
- For long videos (>10 sec), split into ~6 second clips

## Quick Example

```bash
# Add your video
cp my_dance_video.mp4 data/videos/

# Process it
python src/pipeline/make_warped_noise.py \
    data/videos/my_dance_video.mp4 \
    results/warped_noise/my_dance
```

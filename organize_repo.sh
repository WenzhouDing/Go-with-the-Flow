#!/bin/bash
# Organize repository structure

# Move documentation to docs/
mv MOTION_TRANSFER_DEMO.md docs/guides/ 2>/dev/null
mv TEST_MOTION_TRANSFER.md docs/guides/ 2>/dev/null
mv COMPLETE_FFT_DEMO_GUIDE.md docs/guides/ 2>/dev/null
mv QUICK_START_FREQUENCY.md docs/ 2>/dev/null
mv FREQUENCY_MOTION_IMPLEMENTATION.md docs/ 2>/dev/null
mv PIPELINE_SUMMARY.txt docs/ 2>/dev/null
mv IMPLEMENTATION_SUMMARY.txt docs/ 2>/dev/null
mv REPOSITORY_ORGANIZATION.md docs/ 2>/dev/null

# Create scripts directory
mkdir -p scripts

# Move demo scripts
mv demo_motion_transfer.sh scripts/ 2>/dev/null
mv run_fft_demo.sh scripts/ 2>/dev/null

# Keep these in root:
# - Motion_FFT.md (design spec)
# - README.md (main readme)
# - extract_flow_sequence.py (main tool)
# - frequency_motion_transfer.py (main tool)
# - test_frequency_decomposition.py (main tool)

echo "Repository organized!"

"""
QC overlay movie from raw zarr data (full resolution).
Reference (green) + each frame (magenta). Overlap = white.
Uses middle z-slice from the test zarr.
"""

import numpy as np
import zarr
import subprocess

zarr_path = "./results/2026-03-02-01_test10.zarr"
movie_path = "./results/test10_overlay.mp4"
z_slice = 40  # middle slice
channel = 1   # membrane / PinkyCaMP

g = zarr.open(zarr_path, mode="r")
data = g["data"]
n_stacks = data.shape[0]
print(f"Zarr shape: {data.shape}, making movie from {n_stacks} stacks, z={z_slice}, ch={channel}")

# Compute reference as mean of all stacks at this z-slice
print("Computing reference (mean across timepoints)...")
ref_acc = np.zeros((data.shape[3], data.shape[4]), dtype=np.float64)
for t in range(n_stacks):
    ref_acc += np.asarray(data[t, channel, z_slice, :, :]).astype(np.float64)
ref = (ref_acc / n_stacks).astype(np.float32)

# Contrast limits from reference
p1, p99 = np.percentile(ref, [1, 99.5])
print(f"Contrast: {p1:.0f} - {p99:.0f}")
ref_norm = np.clip((ref - p1) / (p99 - p1), 0, 1)

h, w = ref_norm.shape
# Pad to even dims for h264
h_pad = h + (h % 2)
w_pad = w + (w % 2)

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{w_pad}x{h_pad}",
    "-pix_fmt", "rgb24",
    "-r", "2",
    "-i", "-",
    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    movie_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

for t in range(n_stacks):
    frame = np.asarray(data[t, channel, z_slice, :, :]).astype(np.float32)
    frame_norm = np.clip((frame - p1) / (p99 - p1), 0, 1)

    # Green = reference, Magenta = current frame
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = frame_norm    # R
    rgb[:, :, 1] = ref_norm      # G (reference)
    rgb[:, :, 2] = frame_norm    # B

    padded = np.zeros((h_pad, w_pad, 3), dtype=np.uint8)
    padded[:h, :w] = (rgb * 255).astype(np.uint8)
    proc.stdin.write(padded.tobytes())
    print(f"Frame {t+1}/{n_stacks}")

proc.stdin.close()
proc.wait()
print(f"Done! Saved to {movie_path} ({h}x{w}, {n_stacks} frames)")

"""
Generate Z-MIP timeseries movies from unregistered zarr data.
One movie per channel, at 4x real-time speed (~2 min playback).
Used to assess raw motion before registration.
"""

import numpy as np
import zarr
from pathlib import Path
import subprocess

# ── Configuration ──
zarr_path = "/home/lauren/wholistic_preprocessing/preprocessed/2026-03-03-05.zarr"
qc_dir = Path("./ProcessedData/2026-03-03-05/qc")
qc_dir.mkdir(parents=True, exist_ok=True)

gamma = 0.5
fps = 7  # ~2 min playback for 800 frames (4x real-time at 1.7 Hz)

channel_names = {0: "calcium", 1: "membrane"}

# ── Load zarr ──
arr = zarr.open(zarr_path, mode="r")
n_t, n_c, n_z, n_y, n_x = arr.shape
print(f"Zarr shape: {arr.shape}, dtype: {arr.dtype}")

# Pad dimensions to even for h264
h_pad = n_y + (n_y % 2)
w_pad = n_x + (n_x % 2)

for ch_idx, ch_name in channel_names.items():
    print(f"\n{'='*50}")
    print(f"  Channel: {ch_name} (index {ch_idx})")
    print(f"{'='*50}")

    # Compute contrast from sampled MIPs
    sample_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    all_vals = []
    for si in sample_indices:
        vol = np.asarray(arr[si, ch_idx])  # (Z, Y, X)
        mip = vol.max(axis=0)  # (Y, X)
        all_vals.append(mip.ravel())
    all_vals = np.concatenate(all_vals)
    p1, p99 = np.percentile(all_vals, [0.5, 99.9])
    print(f"  Contrast: {p1:.0f} - {p99:.0f}, gamma={gamma}")

    # Open ffmpeg
    out_path = str(qc_dir / f"raw_mip_{ch_name}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_pad}x{h_pad}",
        "-pix_fmt", "gray",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for t in range(n_t):
        vol = np.asarray(arr[t, ch_idx])  # (Z, Y, X)
        mip = vol.max(axis=0).astype(np.float32)  # (Y, X)

        # Normalize with gamma
        norm = np.clip((mip - p1) / max(p99 - p1, 1e-8), 0, 1)
        frame_u8 = (np.power(norm, gamma) * 255).astype(np.uint8)

        # Pad to even dimensions
        padded = np.zeros((h_pad, w_pad), dtype=np.uint8)
        padded[:n_y, :n_x] = frame_u8
        proc.stdin.write(padded.tobytes())

        if (t + 1) % 100 == 0:
            print(f"    Frame {t + 1}/{n_t}")

    proc.stdin.close()
    proc.wait()
    print(f"  Saved: {out_path}")

print("\n=== Done! ===")
for ch_idx, ch_name in channel_names.items():
    print(f"  {ch_name}: {qc_dir}/raw_mip_{ch_name}.mp4")

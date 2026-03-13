"""
Movie of the actual registered data (grayscale) at middle z-slice.
Shows the PinkyCaMP signal as it looks after registration.
"""

import numpy as np
import tifffile
from pathlib import Path
import subprocess

output_dir = Path("./results/2026-03-02-01_test100_registered_2x")
movie_path = "./results/test100_registered.mp4"
stills_dir = Path("./results/test100_registered_stills")
stills_dir.mkdir(exist_ok=True)

z_slice = 40

frame_files = sorted((output_dir / "membrane").glob("frame_*.ome.tif"))
n_frames = len(frame_files)

# Get shape and compute contrast from all frames
sample = tifffile.imread(frame_files[0])
n_z, n_y, n_x = sample.shape
print(f"{n_frames} frames, shape: ({n_z}, {n_y}, {n_x})")

# Compute contrast limits from a few frames
slices = []
for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]:
    slices.append(tifffile.imread(frame_files[i])[z_slice])
all_vals = np.concatenate([s.ravel() for s in slices])
p1, p99 = np.percentile(all_vals, [0.5, 99.9])
print(f"Contrast: {p1:.0f} - {p99:.0f}")

gamma = 0.5  # <1 brightens darks, compresses brights — preserves neuron detail

# Pad to even dims
h_pad = n_y + (n_y % 2)
w_pad = n_x + (n_x % 2)

# --- Movie ---
cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{w_pad}x{h_pad}",
    "-pix_fmt", "gray",
    "-r", "10",
    "-i", "-",
    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    movie_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

for i, fpath in enumerate(frame_files):
    vol = tifffile.imread(fpath)
    sl = vol[z_slice].astype(np.float32)
    norm = np.clip((sl - p1) / (p99 - p1), 0, 1)
    norm = np.power(norm, gamma)  # gamma correction
    frame_u8 = (norm * 255).astype(np.uint8)
    padded = np.zeros((h_pad, w_pad), dtype=np.uint8)
    padded[:n_y, :n_x] = frame_u8
    proc.stdin.write(padded.tobytes())
    if (i + 1) % 20 == 0:
        print(f"  Movie frame {i + 1}/{n_frames}")

proc.stdin.close()
proc.wait()
print(f"Saved movie: {movie_path}")

# --- Stills: XY, YZ, XZ at several timepoints ---
print("\nGenerating stills...")
timepoints = [0, 24, 49, 74, 99]

for t in timepoints:
    vol = tifffile.imread(frame_files[t]).astype(np.float32)

    for view_name, sl in [
        (f"XY_z{z_slice}", vol[z_slice, :, :]),
        (f"YZ_x{n_x // 2}", vol[:, :, n_x // 2]),
        (f"XZ_y{n_y // 2}", vol[:, n_y // 2, :]),
    ]:
        norm = np.clip((sl - p1) / (p99 - p1), 0, 1)
        norm = np.power(norm, gamma)  # gamma correction
        tifffile.imwrite(
            str(stills_dir / f"t{t:03d}_{view_name}.tif"),
            (norm * 255).astype(np.uint8),
        )

    print(f"  t={t}: saved XY, YZ, XZ")

print(f"Done! Stills in {stills_dir}/")

"""
Generate QC outputs for the 100-stack test registration:
1. Overlay movie (green=reference, magenta=registered frame) at z=40
2. Orthogonal still frames (XY, YZ, XZ) at several timepoints
"""

import numpy as np
import tifffile
from pathlib import Path
import subprocess

output_dir = Path("./results/2026-03-02-01_test100_registered_2x")
movie_path = "./results/test100_overlay_2x.mp4"
stills_dir = Path("./results/test100_stills_2x")
stills_dir.mkdir(exist_ok=True)

z_slice = 40   # middle z for XY view
# y and x slices set after reading frame shape (may be downsampled)

# Load all registered membrane frames and reference
frame_files = sorted((output_dir / "membrane").glob("frame_*.ome.tif"))
ref_files = sorted((output_dir / "reference").glob("ref_*.ome.tif"))
n_frames = len(frame_files)
print(f"Found {n_frames} registered frames, {len(ref_files)} reference(s)")

# Load first frame to get shape
sample = tifffile.imread(frame_files[0])
n_z, n_y, n_x = sample.shape
y_slice = n_y // 2
x_slice = n_x // 2
print(f"Frame shape: ({n_z}, {n_y}, {n_x}), slices: z={z_slice}, y={y_slice}, x={x_slice}")

# Use the first (middle block) reference
ref = tifffile.imread(ref_files[0]).astype(np.float32)

# Contrast from reference
p1, p99 = np.percentile(ref, [1, 99.5])
print(f"Contrast: {p1:.0f} - {p99:.0f}")


def normalize(img):
    return np.clip((img.astype(np.float32) - p1) / (p99 - p1), 0, 1)


def make_overlay(ref_slice, frame_slice):
    """Green=reference, Magenta=frame, overlap=white."""
    r = normalize(ref_slice)
    f = normalize(frame_slice)
    h, w = r.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = f  # R (magenta)
    rgb[:, :, 1] = r  # G (reference)
    rgb[:, :, 2] = f  # B (magenta)
    return (rgb * 255).astype(np.uint8)


def pad_even(img):
    h, w = img.shape[:2]
    h_pad = h + (h % 2)
    w_pad = w + (w % 2)
    if h_pad == h and w_pad == w:
        return img
    padded = np.zeros((h_pad, w_pad, *img.shape[2:]), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


# ── 1. Overlay movie (XY at z=40) ──
print("\n=== Generating overlay movie ===")
ref_xy = ref[z_slice]

first_overlay = pad_even(make_overlay(ref_xy, tifffile.imread(frame_files[0])[z_slice]))
h_pad, w_pad = first_overlay.shape[:2]

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{w_pad}x{h_pad}",
    "-pix_fmt", "rgb24",
    "-r", "10",  # 10 fps → 10 seconds for 100 frames
    "-i", "-",
    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    movie_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

for i, fpath in enumerate(frame_files):
    vol = tifffile.imread(fpath)
    overlay = pad_even(make_overlay(ref_xy, vol[z_slice]))
    proc.stdin.write(overlay.tobytes())
    if (i + 1) % 20 == 0:
        print(f"  Movie frame {i + 1}/{n_frames}")

proc.stdin.close()
proc.wait()
print(f"  Saved movie: {movie_path}")

# ── 2. Orthogonal stills ──
print("\n=== Generating orthogonal stills ===")
timepoints = [0, 24, 49, 74, 99]  # early, quarter, middle, three-quarter, late

for t in timepoints:
    vol = tifffile.imread(frame_files[t]).astype(np.float32)
    print(f"\n  t={t}:")

    # XY (axial) at z=40
    xy_overlay = make_overlay(ref[z_slice, :, :], vol[z_slice, :, :])
    tifffile.imwrite(str(stills_dir / f"t{t:03d}_XY_z{z_slice}.tif"), xy_overlay)
    print(f"    XY saved (z={z_slice})")

    # YZ (sagittal) at x=483
    yz_ref = ref[:, :, x_slice]  # (Z, Y)
    yz_frame = vol[:, :, x_slice]
    yz_overlay = make_overlay(yz_ref, yz_frame)
    tifffile.imwrite(str(stills_dir / f"t{t:03d}_YZ_x{x_slice}.tif"), yz_overlay)
    print(f"    YZ saved (x={x_slice})")

    # XZ (coronal) at y=315
    xz_ref = ref[:, y_slice, :]  # (Z, X)
    xz_frame = vol[:, y_slice, :]
    xz_overlay = make_overlay(xz_ref, xz_frame)
    tifffile.imwrite(str(stills_dir / f"t{t:03d}_XZ_y{y_slice}.tif"), xz_overlay)
    print(f"    XZ saved (y={y_slice})")

print(f"\nDone! Stills in {stills_dir}/")

"""
Generate QC outputs for ProcessedData/2026-03-02-01:
1. Overlay movie (green=reference, magenta=registered frame) at z=40
2. Grayscale registered movie with gamma correction
3. Orthogonal stills at several timepoints
4. Rotating 3D MIP GIF at one timepoint
"""

import numpy as np
import tifffile
from pathlib import Path
import subprocess
from scipy.ndimage import rotate, zoom
from PIL import Image

# ── Configuration ──
input_dir = Path("./ProcessedData/2026-03-02-01")
out_prefix = "processed_2026-03-02-01"
out_dir = Path(f"./ProcessedData/2026-03-02-01/qc")
out_dir.mkdir(exist_ok=True)

overlay_movie_path = str(out_dir / f"{out_prefix}_overlay.mp4")
registered_movie_path = str(out_dir / f"{out_prefix}_registered.mp4")
stills_dir = out_dir / "stills_overlay"
registered_stills_dir = out_dir / "stills_registered"
gif_path = str(out_dir / f"{out_prefix}_rotating_mip.gif")
stills_dir.mkdir(exist_ok=True)
registered_stills_dir.mkdir(exist_ok=True)

z_slice = 40
gamma = 0.5

# Load all registered membrane frames and references
frame_files = sorted((input_dir / "membrane").glob("frame_*.ome.tif"))
ref_files = sorted((input_dir / "reference").glob("ref_*.ome.tif"))
n_frames = len(frame_files)
print(f"Found {n_frames} registered frames, {len(ref_files)} reference(s)")

# Load first frame to get shape
sample = tifffile.imread(frame_files[0])
n_z, n_y, n_x = sample.shape
y_slice = n_y // 2
x_slice = n_x // 2
print(f"Frame shape: ({n_z}, {n_y}, {n_x}), slices: z={z_slice}, y={y_slice}, x={x_slice}")

# Use the middle block reference
ref = tifffile.imread(ref_files[len(ref_files) // 2]).astype(np.float32)

# Contrast from sampling several frames
sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
all_vals = np.concatenate([tifffile.imread(frame_files[i])[z_slice].ravel() for i in sample_indices])
p1, p99 = np.percentile(all_vals, [0.5, 99.9])
print(f"Contrast: {p1:.0f} - {p99:.0f}, gamma={gamma}")


def normalize(img):
    norm = np.clip((img.astype(np.float32) - p1) / (p99 - p1), 0, 1)
    return np.power(norm, gamma)


def normalize_nogamma(img):
    return np.clip((img.astype(np.float32) - p1) / (p99 - p1), 0, 1)


def make_overlay(ref_slice, frame_slice):
    """Green=reference, Magenta=frame, overlap=white."""
    r = normalize_nogamma(ref_slice)
    f = normalize_nogamma(frame_slice)
    h, w = r.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :, 0] = f
    rgb[:, :, 1] = r
    rgb[:, :, 2] = f
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
print("\n=== 1. Generating overlay movie ===")
ref_xy = ref[z_slice]
first_overlay = pad_even(make_overlay(ref_xy, tifffile.imread(frame_files[0])[z_slice]))
h_pad, w_pad = first_overlay.shape[:2]

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{w_pad}x{h_pad}",
    "-pix_fmt", "rgb24",
    "-r", "30",
    "-i", "-",
    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    overlay_movie_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

for i, fpath in enumerate(frame_files):
    vol = tifffile.imread(fpath)
    overlay = pad_even(make_overlay(ref_xy, vol[z_slice]))
    proc.stdin.write(overlay.tobytes())
    if (i + 1) % 100 == 0:
        print(f"  Overlay frame {i + 1}/{n_frames}")

proc.stdin.close()
proc.wait()
print(f"  Saved: {overlay_movie_path}")

# ── 2. Grayscale registered movie ──
print("\n=== 2. Generating grayscale registered movie ===")
h_pad_g = n_y + (n_y % 2)
w_pad_g = n_x + (n_x % 2)

cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{w_pad_g}x{h_pad_g}",
    "-pix_fmt", "gray",
    "-r", "30",
    "-i", "-",
    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    registered_movie_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

for i, fpath in enumerate(frame_files):
    vol = tifffile.imread(fpath)
    sl = vol[z_slice].astype(np.float32)
    frame_u8 = (normalize(sl) * 255).astype(np.uint8)
    padded = np.zeros((h_pad_g, w_pad_g), dtype=np.uint8)
    padded[:n_y, :n_x] = frame_u8
    proc.stdin.write(padded.tobytes())
    if (i + 1) % 100 == 0:
        print(f"  Registered frame {i + 1}/{n_frames}")

proc.stdin.close()
proc.wait()
print(f"  Saved: {registered_movie_path}")

# ── 3. Orthogonal stills ──
print("\n=== 3. Generating orthogonal stills ===")
timepoints = [0, 99, 199, 399, 599, 799]

for t in timepoints:
    if t >= n_frames:
        continue
    vol = tifffile.imread(frame_files[t]).astype(np.float32)

    # Overlay stills
    xy_overlay = make_overlay(ref[z_slice, :, :], vol[z_slice, :, :])
    tifffile.imwrite(str(stills_dir / f"t{t:04d}_XY_z{z_slice}.tif"), xy_overlay)

    yz_overlay = make_overlay(ref[:, :, x_slice], vol[:, :, x_slice])
    tifffile.imwrite(str(stills_dir / f"t{t:04d}_YZ_x{x_slice}.tif"), yz_overlay)

    xz_overlay = make_overlay(ref[:, y_slice, :], vol[:, y_slice, :])
    tifffile.imwrite(str(stills_dir / f"t{t:04d}_XZ_y{y_slice}.tif"), xz_overlay)

    # Grayscale registered stills
    for view_name, sl in [
        (f"XY_z{z_slice}", vol[z_slice, :, :]),
        (f"YZ_x{x_slice}", vol[:, :, x_slice]),
        (f"XZ_y{y_slice}", vol[:, y_slice, :]),
    ]:
        tifffile.imwrite(
            str(registered_stills_dir / f"t{t:04d}_{view_name}.tif"),
            (normalize(sl) * 255).astype(np.uint8),
        )

    print(f"  t={t}: saved overlay + registered stills (XY, YZ, XZ)")

# ── 4. Rotating 3D MIP GIF ──
print("\n=== 4. Generating rotating 3D MIP GIF ===")
mip_timepoint = min(399, n_frames - 1)
frame_vol = tifffile.imread(frame_files[mip_timepoint]).astype(np.float32)

# Anisotropy correction: Z=1.0µm, XY=0.386µm (full resolution)
z_scale = 1.0 / 0.386
frame_vol = zoom(frame_vol, (z_scale, 1, 1), order=1)
ref_vol = zoom(ref, (z_scale, 1, 1), order=1)
print(f"  After Z correction: {frame_vol.shape}")

n_angles = 120
angles = np.linspace(0, 360, n_angles, endpoint=False)
gif_frames = []

for i, angle in enumerate(angles):
    rot_frame = rotate(frame_vol, angle, axes=(0, 2), reshape=False, order=1)
    rot_ref = rotate(ref_vol, angle, axes=(0, 2), reshape=False, order=1)

    mip_frame = normalize(rot_frame.max(axis=0))
    mip_ref = normalize(rot_ref.max(axis=0))

    h, w = mip_frame.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (mip_frame * 255).astype(np.uint8)
    rgb[:, :, 1] = (mip_ref * 255).astype(np.uint8)
    rgb[:, :, 2] = (mip_frame * 255).astype(np.uint8)

    gif_frames.append(Image.fromarray(rgb))

    if (i + 1) % 30 == 0:
        print(f"  MIP frame {i + 1}/{n_angles}")

gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=50,
    loop=0,
)
print(f"  Saved: {gif_path}")

print("\n=== All QC outputs complete! ===")
print(f"  Overlay movie:    {overlay_movie_path}")
print(f"  Registered movie: {registered_movie_path}")
print(f"  Overlay stills:   {stills_dir}/")
print(f"  Registered stills:{registered_stills_dir}/")
print(f"  3D MIP GIF:       {gif_path}")

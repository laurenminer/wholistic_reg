"""
Generate QC outputs for dual-channel registered data:
- One overlay movie per channel (green=reference, magenta=registered)
- One rotating 3D MIP GIF per channel
- Diagnostic stats (intensity + correlation) per channel
"""

import numpy as np
import tifffile
from pathlib import Path
import subprocess
from scipy.ndimage import rotate, zoom
from PIL import Image

# ── Configuration ──
input_dir = Path("./ProcessedData/2026-03-03-01")
qc_dir = input_dir / "qc"
qc_dir.mkdir(exist_ok=True)

z_slice = 40
gamma = 0.5

# Anisotropy correction: Z-step=0.54µm, XY=1.158µm (0.386µm * 3x lateral bin)
z_scale = 0.54 / 1.158

channels = ["membrane", "calcium"]

# Load first frame to get shape
sample = tifffile.imread(sorted((input_dir / "membrane").glob("frame_*.ome.tif"))[0])
n_z, n_y, n_x = sample.shape
print(f"Frame shape: ({n_z}, {n_y}, {n_x})")

# Use the middle block reference
ref_files = sorted((input_dir / "reference").glob("ref_*.ome.tif"))
ref = tifffile.imread(ref_files[len(ref_files) // 2]).astype(np.float32)

# Reference normalization (independent of frame data)
ref_p1, ref_p99 = np.percentile(ref, [0.5, 99.9])
print(f"Reference contrast: {ref_p1:.1f} - {ref_p99:.1f}")


def normalize_ref(img):
    norm = np.clip((img.astype(np.float32) - ref_p1) / max(ref_p99 - ref_p1, 1e-8), 0, 1)
    return np.power(norm, gamma)


def normalize_ref_nogamma(img):
    return np.clip((img.astype(np.float32) - ref_p1) / max(ref_p99 - ref_p1, 1e-8), 0, 1)


def pad_even(img):
    h, w = img.shape[:2]
    h_pad = h + (h % 2)
    w_pad = w + (w % 2)
    if h_pad == h and w_pad == w:
        return img
    padded = np.zeros((h_pad, w_pad, *img.shape[2:]), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


for ch in channels:
    print(f"\n{'='*60}")
    print(f"  Processing channel: {ch}")
    print(f"{'='*60}")

    frame_files = sorted((input_dir / ch).glob("frame_*.ome.tif"))
    n_frames = len(frame_files)
    print(f"  Found {n_frames} frames")

    # Compute contrast for this channel (independent of reference)
    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    all_vals = np.concatenate([
        tifffile.imread(frame_files[i])[z_slice].ravel() for i in sample_indices
    ])
    p1, p99 = np.percentile(all_vals, [0.5, 99.9])
    print(f"  Frame contrast: {p1:.1f} - {p99:.1f}, gamma={gamma}")

    def normalize_frame(img, _p1=p1, _p99=p99):
        norm = np.clip((img.astype(np.float32) - _p1) / max(_p99 - _p1, 1e-8), 0, 1)
        return np.power(norm, gamma)

    def normalize_frame_nogamma(img, _p1=p1, _p99=p99):
        return np.clip((img.astype(np.float32) - _p1) / max(_p99 - _p1, 1e-8), 0, 1)

    def make_overlay(ref_slice, frame_slice):
        """Green=reference, Magenta=frame, overlap=white."""
        r = normalize_ref_nogamma(ref_slice)
        f = normalize_frame_nogamma(frame_slice)
        h, w = r.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, 0] = f
        rgb[:, :, 1] = r
        rgb[:, :, 2] = f
        return (rgb * 255).astype(np.uint8)

    # ── Diagnostic stats ──
    print(f"\n  --- Diagnostic Stats ({ch}) ---")
    print(f"  Reference: mean={ref.mean():.2f}, std={ref.std():.2f}, "
          f"range=[{ref.min():.1f}, {ref.max():.1f}], "
          f"percentiles=[{ref_p1:.1f}, {ref_p99:.1f}]")

    for si in [0, n_frames // 2, n_frames - 1]:
        frame = tifffile.imread(frame_files[si]).astype(np.float32)
        ref_flat = ref[z_slice].ravel()
        frame_flat = frame[z_slice].ravel()
        r_norm = (ref_flat - ref_flat.mean()) / (ref_flat.std() + 1e-8)
        f_norm = (frame_flat - frame_flat.mean()) / (frame_flat.std() + 1e-8)
        corr = np.mean(r_norm * f_norm)
        print(f"  Frame {si:>4d}: mean={frame.mean():.2f}, "
              f"range=[{frame.min():.1f}, {frame.max():.1f}], "
              f"corr(z={z_slice})={corr:.4f}")

    print(f"  Intensity ratio (ref/frame): {ref.mean() / (all_vals.mean() + 1e-8):.1f}x")

    # ── 1. Overlay movie ──
    print(f"\n  --- Overlay movie ({ch}) ---")
    overlay_path = str(qc_dir / f"overlay_{ch}.mp4")
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
        overlay_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i, fpath in enumerate(frame_files):
        vol = tifffile.imread(fpath)
        overlay = pad_even(make_overlay(ref_xy, vol[z_slice]))
        proc.stdin.write(overlay.tobytes())
        if (i + 1) % 100 == 0:
            print(f"    Frame {i + 1}/{n_frames}")

    proc.stdin.close()
    proc.wait()
    print(f"    Saved: {overlay_path}")

    # ── 2. Rotating 3D MIP GIF ──
    print(f"\n  --- 3D MIP GIF ({ch}) ---")
    gif_path = str(qc_dir / f"rotating_mip_{ch}.gif")
    mip_timepoint = min(399, n_frames - 1)
    frame_vol = tifffile.imread(frame_files[mip_timepoint]).astype(np.float32)

    frame_vol_scaled = zoom(frame_vol, (z_scale, 1, 1), order=1)
    ref_vol_scaled = zoom(ref, (z_scale, 1, 1), order=1)
    print(f"    After Z correction: {frame_vol_scaled.shape}")

    n_angles = 120
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    gif_frames = []

    for i, angle in enumerate(angles):
        rot_frame = rotate(frame_vol_scaled, angle, axes=(0, 2), reshape=False, order=1)
        rot_ref = rotate(ref_vol_scaled, angle, axes=(0, 2), reshape=False, order=1)

        mip_frame = normalize_frame(rot_frame.max(axis=0))
        mip_ref = normalize_ref(rot_ref.max(axis=0))

        h, w = mip_frame.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (mip_frame * 255).astype(np.uint8)
        rgb[:, :, 1] = (mip_ref * 255).astype(np.uint8)
        rgb[:, :, 2] = (mip_frame * 255).astype(np.uint8)

        gif_frames.append(Image.fromarray(rgb))

        if (i + 1) % 30 == 0:
            print(f"    MIP frame {i + 1}/{n_angles}")

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=50,
        loop=0,
    )
    print(f"    Saved: {gif_path}")

print("\n=== All QC outputs complete! ===")
for ch in channels:
    print(f"  {ch}: overlay → qc/overlay_{ch}.mp4, MIP → qc/rotating_mip_{ch}.gif")

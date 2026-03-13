"""
Rotating 3D maximum intensity projection (MIP) GIF.
Shows overlay of reference (green) and registered frame (magenta) rotating 360°.
"""

import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import rotate, zoom
from PIL import Image

output_dir = Path("./results/2026-03-02-01_test100_registered_2x")
gif_path = "./results/test100_rotating_mip.gif"
timepoint = 49

# Load data
frame = tifffile.imread(output_dir / "membrane" / f"frame_{timepoint:06d}.ome.tif").astype(np.float32)
ref_files = sorted((output_dir / "reference").glob("ref_*.ome.tif"))
ref = tifffile.imread(ref_files[0]).astype(np.float32)
print(f"Frame shape: {frame.shape}, Reference shape: {ref.shape}")

# Anisotropy correction: Z=1.0µm, XY=0.772µm (2x downsample) → zoom Z by 1.3×
z_scale = 1.0 / 0.772
print(f"Z scale factor: {z_scale:.2f}")
frame = zoom(frame, (z_scale, 1, 1), order=1)
ref = zoom(ref, (z_scale, 1, 1), order=1)
print(f"After Z correction: {frame.shape}")

# Contrast from reference
p1, p99 = np.percentile(ref, [0.5, 99.9])
gamma = 0.5
print(f"Contrast: {p1:.0f} - {p99:.0f}, gamma={gamma}")


def normalize(img):
    norm = np.clip((img - p1) / (p99 - p1), 0, 1)
    return np.power(norm, gamma)


# Generate rotating MIP frames
n_angles = 120
angles = np.linspace(0, 360, n_angles, endpoint=False)
frames = []

for i, angle in enumerate(angles):
    # Rotate both volumes around Y axis (axes 0=Z, 2=X)
    rot_frame = rotate(frame, angle, axes=(0, 2), reshape=False, order=1)
    rot_ref = rotate(ref, angle, axes=(0, 2), reshape=False, order=1)

    # Max intensity projection along axis 0 (viewing axis after rotation)
    mip_frame = normalize(rot_frame.max(axis=0))
    mip_ref = normalize(rot_ref.max(axis=0))

    # Overlay: green=reference, magenta=registered
    h, w = mip_frame.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (mip_frame * 255).astype(np.uint8)  # R (magenta)
    rgb[:, :, 1] = (mip_ref * 255).astype(np.uint8)    # G (reference)
    rgb[:, :, 2] = (mip_frame * 255).astype(np.uint8)  # B (magenta)

    frames.append(Image.fromarray(rgb))

    if (i + 1) % 20 == 0:
        print(f"  Frame {i + 1}/{n_angles}")

# Save GIF
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # ms per frame → 6s loop
    loop=0,
)
print(f"Saved GIF: {gif_path}")

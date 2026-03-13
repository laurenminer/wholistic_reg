"""
Preprocess ND2 file into a properly-structured Zarr for 3D registration.

The ND2 reader treats the file as 64,000 individual 2D frames,
but the data is actually 800 z-stacks x 80 z-slices x 2 channels.
This script reshapes it into (T, C, Z, Y, X) = (800, 2, 80, 630, 966).
"""

import nd2
import zarr
import zarr.codecs
import numpy as np

nd2_path = "/store1/lauren/Tetramisole_Immobilized_Imaging/2026_PinkyCamp_Immobilized/data_raw/2026-03-02-01.nd2"
zarr_path = "./results/2026-03-02-01.zarr"

z_per_stack = 80
n_channels = 2

f = nd2.ND2File(nd2_path)
total_frames = f.sizes.get("T", 1) * f.sizes.get("Z", 1)
print(f"ND2 sizes: {dict(f.sizes)}")
print(f"Total frames from metadata: {total_frames}")

# Determine actual frame count and dimensions from first frame
first_frame = np.asarray(f.read_frame(0))
print(f"First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")

if first_frame.ndim == 3:
    # (C, Y, X)
    n_y, n_x = first_frame.shape[1], first_frame.shape[2]
    frame_channels = first_frame.shape[0]
elif first_frame.ndim == 2:
    # (Y, X)
    n_y, n_x = first_frame.shape
    frame_channels = 1
else:
    raise ValueError(f"Unexpected frame shape: {first_frame.shape}")

# Use actual frame count
n_stacks = total_frames // z_per_stack
print(f"Reshaping into {n_stacks} z-stacks x {z_per_stack} z-slices x {frame_channels} channels")
print(f"Output shape: ({n_stacks}, {frame_channels}, {z_per_stack}, {n_y}, {n_x})")

# Create Zarr store with shape (T, C, Z, Y, X)
store = zarr.open_group(zarr_path, mode="w")
data = store.create_array(
    "data",
    shape=(n_stacks, frame_channels, z_per_stack, n_y, n_x),
    chunks=(1, frame_channels, z_per_stack, n_y, n_x),
    dtype=first_frame.dtype,
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=1),
)

# Read frame-by-frame from ND2 and reshape into z-stacks
for stack_idx in range(n_stacks):
    for z in range(z_per_stack):
        frame_idx = stack_idx * z_per_stack + z
        frame = np.asarray(f.read_frame(frame_idx))
        if frame.ndim == 3:
            data[stack_idx, :, z, :, :] = frame
        else:
            data[stack_idx, 0, z, :, :] = frame

    if stack_idx % 50 == 0:
        print(f"Stack {stack_idx}/{n_stacks} ({100*stack_idx/n_stacks:.0f}%)")

f.close()
print(f"Done! Saved Zarr to {zarr_path}")
print(f"Final shape: {data.shape}")

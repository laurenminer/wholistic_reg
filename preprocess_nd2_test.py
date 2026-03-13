"""
Create a small test Zarr with 10 z-stacks for validating 3D registration.

Reads first 800 frames (10 stacks × 80 z-slices) from the ND2 file
and writes to (T, C, Z, Y, X) = (10, 2, 80, 630, 966).
"""

import nd2
import zarr
import zarr.codecs
import numpy as np

nd2_path = "/store1/lauren/Tetramisole_Immobilized_Imaging/2026_PinkyCamp_Immobilized/data_raw/2026-03-02-01.nd2"
zarr_path = "./results/2026-03-02-01_test10.zarr"

z_per_stack = 80
n_stacks = 10  # just 10 stacks for testing

f = nd2.ND2File(nd2_path)
first_frame = np.asarray(f.read_frame(0))
print(f"First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")

n_channels = first_frame.shape[0]  # 2
n_y, n_x = first_frame.shape[1], first_frame.shape[2]

print(f"Output shape: ({n_stacks}, {n_channels}, {z_per_stack}, {n_y}, {n_x})")

store = zarr.open_group(zarr_path, mode="w")
data = store.create_array(
    "data",
    shape=(n_stacks, n_channels, z_per_stack, n_y, n_x),
    chunks=(1, n_channels, z_per_stack, n_y, n_x),
    dtype=first_frame.dtype,
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=1),
)

for stack_idx in range(n_stacks):
    for z in range(z_per_stack):
        frame_idx = stack_idx * z_per_stack + z
        frame = np.asarray(f.read_frame(frame_idx))
        data[stack_idx, :, z, :, :] = frame

    print(f"Stack {stack_idx + 1}/{n_stacks}")

f.close()
print(f"Done! Saved test Zarr to {zarr_path}")
print(f"Final shape: {data.shape}")

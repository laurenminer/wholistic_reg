"""
Create a 100-stack test zarr by copying from the (partially written) full zarr.
Much faster than re-reading from NFS since the data is already local.
"""

import zarr
import zarr.codecs
import numpy as np

src_path = "./results/2026-03-02-01.zarr"
dst_path = "./results/2026-03-02-01_test100.zarr"
n_stacks = 100

src = zarr.open(src_path, mode="r")
src_data = src["data"]
print(f"Source shape: {src_data.shape}, dtype: {src_data.dtype}")

_, n_ch, n_z, n_y, n_x = src_data.shape

dst = zarr.open_group(dst_path, mode="w")
dst_data = dst.create_array(
    "data",
    shape=(n_stacks, n_ch, n_z, n_y, n_x),
    chunks=(1, n_ch, n_z, n_y, n_x),
    dtype=src_data.dtype,
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=1),
)

for t in range(n_stacks):
    dst_data[t] = np.asarray(src_data[t])
    if (t + 1) % 10 == 0:
        print(f"Stack {t + 1}/{n_stacks}")

print(f"Done! Saved to {dst_path}, shape: {dst_data.shape}")

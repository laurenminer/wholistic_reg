from pathlib import Path

import numpy as np
import zarr
import zarrs

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

from wholistic_registration.v2 import RegistrationConfig, RegistrationPipeline
from wholistic_registration.v2.config import (
    BackendConfig,
    ChannelConfig,
    DownsampleConfig,
    MaskConfig,
    PyramidConfig,
    ReferenceConfig,
)
from wholistic_registration.v2.io import BaseReader, Metadata


class FlavellZarrReader(BaseReader):
    def __init__(self, path: str | Path, metadata: Metadata):
        super().__init__(path)
        self._data_arr = np.asarray(zarr.load(str(self.path), mode='r'))
        self._metadata = metadata
    
    @property
    def metadata(self) -> Metadata:
        return self._metadata
    
    def read_frames(
        self,
        frames: list[int],
        channel: int,
        z_slices: list[int] | None = None,
        xy_downsample: int = 1, # Don't change this.
    ) -> np.ndarray:

        if z_slices:
            return self._data_arr[frames, channel, z_slices, ...]
        else:
            return self._data_arr[frames, channel, ...]


ZARR_DATA_PATH = "/path/to/data.zarr"
OUTPUT_DIR = "/path/to/tiff_output/dir"
BIN_FACTOR = 3 # set to bin factor used during preprocessing
GPU_ID = 2 # This could also be set with CUDA_VISIBLE_DEVICES

config = RegistrationConfig(
    input_path=ZARR_DATA_PATH,
    output_dir=OUTPUT_DIR,
    downsample=DownsampleConfig(
        xy=1, # Keep this at 1--we do the binning ourselves in preprocess videos
        z_slices=None,
        t_chunk=20,
    ),
    channels=ChannelConfig(
        dual_channel=False,
        membrane_channel=1,
        calcium_channel=0,
    ),
    reference=ReferenceConfig(
        window_size=40,
        initial_frames=80,
        max_correlation_frames=50,
    ),
    pyramid=PyramidConfig(
        layers=1,
        patch_radius=5,
        iterations=10,
        smooth_penalty=0.08,
    ),
    mask=MaskConfig(
        threshold_factor=5.0,
        intensity_range=(5, 4000),
    ),
    backend=BackendConfig(device="cuda", gpu_id=GPU_ID),
)

metadata = Metadata(
    n_frames = 800,
    n_channels = 2,
    shape_zyx = (80, 660 // BIN_FACTOR, 990 // BIN_FACTOR),
    voxel_size_um = (0.18 * BIN_FACTOR, 0.18 * BIN_FACTOR, 0.18 * BIN_FACTOR),
    frame_rate_hz = 1.7, # not sure if necessary?
    source_format = 'zarr',
    source_path = str(ZARR_DATA_PATH),
    extra = {},
)

pipeline = RegistrationPipeline(config)
reader = FlavellZarrReader(ZARR_DATA_PATH, metadata)

# Substitute our custom reader into the pipeline
pipeline.reader = reader
pipeline.metadata = pipeline.reader.metadata
pipeline.config._metadata = pipeline.metadata.to_dict()

# Run the pipeline
pipeline.run()

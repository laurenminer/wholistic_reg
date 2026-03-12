from wholistic_registration.v2 import RegistrationPipeline, RegistrationConfig
from wholistic_registration.v2.config import (
    DownsampleConfig,
    ChannelConfig,
    ReferenceConfig,
    PyramidConfig,
    MaskConfig,
    BackendConfig,
)

config = RegistrationConfig(
    input_path="/store1/lauren/Tetramisole_Immobilized_Imaging/2026_PinkyCamp_Immobilized/data_raw/2026-03-02-01.nd2",
    output_dir="./results/2026-03-02-01_registered",
    downsample=DownsampleConfig(
        xy=4,
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
    backend=BackendConfig(device="cuda", gpu_id=2),
)

pipeline = RegistrationPipeline(config)
pipeline.run()

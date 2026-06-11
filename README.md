# wholistic_reg

The **registration stage** of the wholistic whole-brain imaging pipeline:
**preprocessing ([wholistic_preprocessing](https://github.com/laurenminer/wholistic_preprocessing))
→ registration → segmentation
([wholistic_segmentation](https://github.com/laurenminer/wholistic_segmentation))**.

Registers preprocessed whole-brain volumes over time and produces QC figures/movies.

## Layout
- `main_template.py` / `main_template_interactive.py` — registration drivers (zarr-backed);
  copy and edit per dataset.
- `wholistic_registration/` — the core registration package (`src/`, its own `pyproject.toml`).
- **QC + visualization scripts**:
  - `make_qc_full.py`, `make_qc_dual.py`, `make_qc_processed.py` — QC figures.
  - `make_registered_movie.py`, `make_overlay_movie.py`, `make_raw_mip_movies.py` — movies.
  - `make_3d_projection.py` — 3D projection render.

## Outputs
Per-dataset results (`ProcessedData/`, `results/`) and large media (`*.tif`, etc.) are
gitignored — code only is tracked here.

## Requirements
See `pyproject.toml` / `wholistic_registration/pyproject.toml`. Managed with `uv`.

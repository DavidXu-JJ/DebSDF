# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

DebSDF is a GPU-dependent ML research project for neural indoor scene reconstruction (SDF-based). It is a single Python package (not a monorepo) with no web frontend, no external services, and no databases. See `README.md` for full setup and usage instructions.

### Key constraints

- **NVIDIA GPU required**: Training, inference, and several core modules (`model/density.py`, `utils/rend_util.py`, `hashencoder/`) call `.cuda()` or require CUDA JIT compilation. The cloud VM does not have a GPU, so full end-to-end training/inference cannot run here.
- **Training data required**: Scene data (ScanNet, Replica, BlenderSwap, TNT) must be downloaded from Google Drive and placed in `./data/`. Without data, even with a GPU, training cannot start.
- **No linter/formatter configured**: The project has no ESLint, flake8, pylint, mypy, or black configuration.
- **No automated test suite**: There are no unit or integration tests.

### What works on CPU (no GPU)

- All Python dependencies install and import correctly (PyTorch CPU, requirements.txt, evaluation_requirements.txt).
- `python3-tk` system package is needed for `model/ray_sampler.py` (imports `tkinter`).
- Config file parsing (all 11 `.conf` files in `code/confs/`).
- CPU-only model components: `model/embedder.py`, `model/loss.py` (e.g. `compute_scale_and_shift`).
- Utilities: `utils/general.py` (e.g. `BackprojectDepth`), `utils/diff_operators.py`, `utils/plots.py`.
- Module loading: `datasets/scene_dataset.py`, `training/debsdf_train.py`, `model/ray_sampler.py`.

### What requires GPU

- `hashencoder/backend.py`: JIT-compiles CUDA kernels via `torch.utils.cpp_extension.load()`. Requires NVCC and a CUDA-capable GPU.
- `model/density.py`: `LaplaceDensity.__init__` calls `.cuda()`.
- `utils/rend_util.py`: `get_camera_params` calls `.cuda()`.
- `utils/general.py`: `get_time()` calls `torch.cuda.synchronize()`.
- Training entry point: `code/training/exp_runner.py` calls `torch.cuda.set_device()` and `torch.distributed.init_process_group(backend='nccl')`.

### Running training (on a GPU machine)

From `README.md`:
```bash
cd code
python -m torch.distributed.launch --master_port 22112 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf ./confs/scannet_mlp.conf --scan_id 1
```

Or use the helper script: `cd code && sh train.sh ./confs/scannet_mlp.conf 1 0`

### Dependencies

- Python 3.10+ (3.12 also works for dependency installation)
- PyTorch 2.2.2 (CPU or CUDA variant depending on environment)
- `requirements.txt` (main deps) and `evaluation_requirements.txt` (open3d, pyrender for evaluation)
- System package: `python3-tk`

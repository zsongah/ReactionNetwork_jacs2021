# FlowER Local Deployment (Apple Silicon, CPU)

This project shells out to FlowER as a subprocess. The FlowER source
and its neural-network weights are **not** committed to this repo
(license + size). You clone them once, into ``vendor/`` (git-ignored),
following the steps below.

## 1. Layout (after setup)

```
ReactionNetwork_jacs2021/
├── flower-env.yml            # conda spec for the FlowER env (committed)
├── vendor/                   # git-ignored; you populate it
│   └── FlowER/               # `git clone https://github.com/FongMunHong/FlowER`
│       ├── beam_predict.py
│       ├── settings.py
│       ├── model/
│       ├── scripts/
│       └── checkpoints/      # weights go here
│           └── flower_new_dataset/
│               └── best_large_hyperparam/
│                   └── model.2880000_95.pt
```

## 2. One-time setup

### 2a. Clone FlowER

```bash
mkdir -p vendor && cd vendor
git clone https://github.com/FongMunHong/FlowER.git
cd ..
```

### 2b. Conda env

```bash
conda env create -f flower-env.yml
conda activate flower
```

This installs `torch==2.4.0`, `rdkit==2024.3.3`, `networkx==3.3`,
`torchdiffeq==0.2.4`, etc. **It is intentionally a separate env from
`jacs2021`** — pinning conflicts with `networkx 3.6 / rdkit 2026.3`.

PyTorch on Apple Silicon will report `cuda=False, mps=True`. FlowER's
`beam_predict.py` falls back to CPU automatically (`device = cuda if
available else cpu`); MPS is *not* used by default because not all
operations in the flow-matching solver have MPS kernels.

### 2b. Download the model checkpoint

1. Open the FlowER Figshare page:
   <https://figshare.com/articles/dataset/FlowER_-_Mechanistic_datasets_and_model_checkpoint/28359407/3>
2. Download **only `checkpoints.zip`** (you do not need `data.zip`
   unless you intend to retrain).
3. Unzip into `vendor/FlowER/`:

   ```bash
   cd vendor/FlowER
   unzip ~/Downloads/checkpoints.zip            # creates checkpoints/
   ls checkpoints/
   # → flower_dataset/  flower_new_dataset/
   ```

4. Sanity check that the expected `.pt` exists:

   ```bash
   ls checkpoints/flower_new_dataset/best_large_hyperparam/model.2880000_95.pt
   # → 186 MB; md5 == 2ab43d1956aafc12ead0b707b2b3c9db (full zip)
   ```

## 3. Smoke-test the FlowER install

From the project root, with the JACS-2021 env active and the FlowER
env vars set per §4:

```bash
conda activate jacs2021
python -m src.flower_smoke_test
```

This runs FlowER on a single canonical reactant set (`EC + H2O`) and
prints the top-K predicted product sets with probabilities. Expect:

* **First call**: ~30–120 s on M-series CPU (model load + beam search).
* **Subsequent calls** with the same reactant: instant (cache hit).

If you see `FlowERUnavailable: FlowER repo or checkpoint not configured`,
re-check `FLOWER_REPO_PATH` / `FLOWER_MODEL_PATH` (see §4).

## 4. Wire FlowER into the JACS-2021 pipeline

The driver script (`src/run_demo.py`) reads three env vars. Note that
`FLOWER_MODEL_PATH` is the **directory containing the .pt file**, not
the file itself — FlowER appends `MODEL_NAME` internally.

```bash
export FLOWER_REPO_PATH="$PWD/vendor/FlowER"
export FLOWER_MODEL_PATH="$PWD/vendor/FlowER/checkpoints/flower_new_dataset/best_large_hyperparam"
export FLOWER_PYTHON_EXECUTABLE="$HOME/miniforge3/envs/flower/bin/python"
```

The default `model_name` in ``FlowERConfig`` is ``model.2880000_95.pt``,
matching the ``flower_new_dataset/best_large_hyperparam`` checkpoint
shipped on Figshare. To use a different checkpoint, also set
``FLOWER_MODEL_NAME`` / ``FLOWER_DATA_NAME`` / ``FLOWER_EXP_NAME``.

Both ``FLOWER_<NAME>`` and bare ``<NAME>`` env-var prefixes are
recognised; ``FLOWER_*`` is preferred to avoid colliding with FlowER's
own ``MODEL_PATH`` etc. when both envs are active in the same shell.

Then run with FlowER as a generator (rather than re-ranker):

```bash
conda activate jacs2021      # main pipeline env
python -m src.run_demo \
    --backend both \
    --flower-as-generator \
    --max-pool-iterations 1 \
    --flower-prob-threshold 0.05 \
    --combo-sizes 1,2 \
    --use-tier-bias \
    --lam 0.5
```

The `jacs2021` env shells out to the `flower` env via the wrapped
subprocess defined in `src/flower_backend.py::_run_subprocess`. You do
**not** need to activate `flower` manually — `flower_backend.py` invokes
`vendor/FlowER/beam_predict.py` with the `flower` env's Python
interpreter (auto-discovered from `~/miniforge3/envs/flower/bin/python`).

## 5. Cache behavior

Every FlowER call is keyed by canonical reactant SMILES + checkpoint
name and persisted under `cache/flower/<sha1>.json`. On a cache hit, no
subprocess is spawned. Recommended workflow on a slow CPU:

1. Run the demo once with `--flower-as-generator`. Even a small pool
   triggers ~50–200 unique reactant combos → ~50–200 FlowER calls →
   30 min – 2 h on first run.
2. Subsequent runs with the same seeds finish in seconds.
3. To pre-warm the cache from a faster machine, `scp -r cache/flower/`
   between hosts.

## 6. Known issues on macOS

* **MPS fallback warning**: `aten::scatter_add_` may print a warning
  about falling back to CPU. Harmless; the operation still runs.
* **First import is slow**: `torch` lazy-loads MKL/OpenMP shims; budget
  10–15 s for the first FlowER subprocess.
* **`Killed: 9`**: macOS killed the process for memory pressure. The
  FlowER large model takes ~6 GB resident on CPU; close Chrome.

## 7. What is NOT supported on Apple Silicon

* GPU acceleration (no NVIDIA → no CUDA).
* `beam_predict_multiGPU.py` (uses `torch.distributed`; single-GPU only
  path is `beam_predict.py`).
* Training (`train.py`) — feasible but excruciating on CPU; use a Linux
  GPU box if you intend to fine-tune.

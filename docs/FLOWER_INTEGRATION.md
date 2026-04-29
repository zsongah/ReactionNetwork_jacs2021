# FlowER integration

This project's species-pool builder can be augmented with **FlowER**
(Joung & Fong et al., *Nature* **645**, 115 (2025);
[github.com/FongMunHong/FlowER](https://github.com/FongMunHong/FlowER)),
an electron-flow generative model for organic reaction mechanisms. FlowER
is invoked as a subprocess from `src/flower_backend.py` so that this
repository remains importable on machines without PyTorch / CUDA.

## What FlowER does and does not cover

FlowER's training set (USPTO + RmechDB + PmechDB) is **organic-only**:

| Reactant property                 | Routed to |
| --------------------------------- | --------- |
| Contains Li / Na / K / Mg / Ca    | fragrec   |
| Net charge ≠ 0                    | fragrec   |
| Open-shell (radical)              | fragrec   |
| Explicit electron transfer step   | fragrec   |
| Otherwise                         | FlowER    |

The split is enforced by `is_organic_combo()`. The two backends are then
**unioned** into a single species pool, so FlowER never *replaces*
fragrec — it augments it.

## How the FlowER prior enters the pathfinding cost

`src/thermo.py::reaction_cost` is

```
cost = softplus(ΔG, scale) − λ · log(max(P_FlowER, p_floor))
```

where `P_FlowER ∈ [0, 1]` is `count / sample_size` from FlowER's beam
search. Two key knobs:

* `λ = 0` reproduces the JACS 2021 paper's original cost exactly (and is
  the default).
* `p_floor = 0.01` is the floor used for reactions FlowER did not
  produce (or that fragrec found in non-organic territory). It prevents
  `−log P_FlowER` from blowing up the cost of perfectly reasonable
  Li-coordination steps.

Pass them at the CLI:

```bash
python -m src.run_demo --backend both --lam 0.5
```

## Setting up FlowER on a GPU host

FlowER needs Linux + ≥25 GB CUDA ≥12.2. The author's pinned environment
is in `FlowER/environment.yml`. Quick recipe:

```bash
git clone https://github.com/FongMunHong/FlowER.git
cd FlowER
conda env create -f environment.yml -n flower
conda activate flower
# Download checkpoints + data (~700 MB total), figshare 28359407
mkdir -p data && cd data && wget <figshare-data.zip-url> && unzip data.zip && cd ..
mkdir -p checkpoints && cd checkpoints && wget <figshare-checkpoints.zip-url> \
    && unzip checkpoints.zip && cd ..
```

Then point this repo at the checkout and the trained weights:

```bash
export FLOWER_REPO_PATH=$HOME/code/FlowER
export FLOWER_MODEL_PATH=$HOME/code/FlowER/checkpoints/best_large_hyperparam
# optional — defaults are usually fine
export FLOWER_MODEL_NAME=model.2880000_95.pt
export FLOWER_SAMPLE_SIZE=50
```

## Running FlowER once and reusing results on a laptop

`src/flower_backend.py` writes a JSON file to `cache/flower/` for every
reactant set it predicts. The filename is a SHA-1 of
`(canonical_dot_smiles, model_name, sample_size)`, so the cache is
deterministic and **portable**.

Workflow:

1. On the GPU host, run `python -m src.run_demo --backend both` (or
   `--backend flower`). FlowER predicts each organic combo once, the
   results land in `cache/flower/*.json`.
2. Copy `cache/flower/` to the same path on your laptop:
   ```bash
   rsync -av gpu-host:~/ReactionNetwork_jacs2021/cache/flower/ ./cache/flower/
   ```
3. On the laptop, `python -m src.run_demo --backend both` will resolve
   every prediction from cache. `is_available()` is allowed to return
   `False` as long as no cache miss occurs; only a miss raises
   `FlowERUnavailable`.

## Cache schema

Each file is a JSON list of `[product_dot_smiles, count]` pairs:

```json
[
  ["CO.OC=O", 25],
  ["O=C(O)OCC", 12]
]
```

`count / sample_size` is the probability assigned to each predicted
product set. The order is FlowER's beam-search ranking, top-`nbest` only
(default 3).

## Troubleshooting

* **"FlowER repo or checkpoint not configured."** — set the two env
  vars; `is_available()` requires both `beam_predict.py` and the model
  directory to exist.
* **Subprocess returns non-zero** — `FlowERUnavailable` carries the last
  2 KB of stderr. Common cause: wrong `EMB_DIM` for the chosen
  checkpoint. The default 256 matches `model.2880000_95.pt`; the smaller
  released models use 128.
* **Cache hit rate < 100% on the laptop** — your seeds or
  `n_frag_steps` differ between hosts. Either run with the same params
  on both, or pre-warm the cache by enumerating the full reactant-combo
  set on the GPU box first.

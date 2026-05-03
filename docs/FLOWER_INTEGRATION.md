# FlowER integration

This project's species-pool builder can be augmented with **FlowER**
(Joung & Fong et al., *Nature* **645**, 115 (2025);
[github.com/FongMunHong/FlowER](https://github.com/FongMunHong/FlowER)),
an electron-flow generative model for organic reaction mechanisms. FlowER
is invoked as a subprocess from `src/flower_backend.py` so that this
repository remains importable on machines without PyTorch / CUDA.

For *running* FlowER on the same Apple-Silicon machine that runs the
JACS-2021 pipeline, see the companion guide
[`FLOWER_LOCAL_DEPLOY.md`](FLOWER_LOCAL_DEPLOY.md). This document focuses
on the **architectural role** FlowER plays in the network and the
controls you have over its behaviour.

## Two modes

| Mode | Flag | What FlowER does |
| ---- | ---- | ---------------- |
| **Re-ranker** (default) | (no flag) | fragrec builds the entire species pool. FlowER is consulted only to attach a probability prior `P_FlowER` to organic reactions that already exist. `λ = 0` exactly reproduces the JACS 2021 cost. |
| **Generator** | `--flower-as-generator` | FlowER **generates new species** from organic combos drawn from the current pool, products are admitted into the pool, the pool is re-enumerated. Iterates `--max-pool-iterations` times. fragrec still runs in parallel and covers Li / charged / radical territory FlowER cannot represent. |

In both modes, fragrec is *never* removed: the architectural blind spots
of FlowER's BE-matrix (no slot for free electrons; oxidation states OOD;
metals 100 % OOD) are exactly what fragrec covers.

## What FlowER does and does not cover

FlowER's training set (USPTO + RmechDB + PmechDB) is organic-only.
``flower_backend.py`` enforces a four-layer routing pipeline before any
subprocess is spawned:

| Layer | Function | Purpose |
| ----- | -------- | ------- |
| 0 | `is_flower_compatible` | Hard architectural gate: no metals, net charge zero, ≤ 30 heavy atoms, elements ⊂ {C, H, N, O, S, P, F, Cl, Br, I}. Open-shell (radical) is allowed — RmechDB contributes ~5K such steps. |
| 1 | `is_reactive_combo` | Cheap heuristic: combo size ≤ 3, total heavy atoms ≤ 25, *some* species must carry a radical / multi-bond / lone-pair-bearing element. Skips inert-pair combos like CH₄ + CH₄ where FlowER would only hallucinate. |
| 2 | `is_acceptable_flower_product` | Post-output guard: each product must itself satisfy Layer 0; product set must conserve heavy-atom multiset within ±1 (FlowER occasionally invents/loses an H during a beam-search miscount). |
| 3 | `_smiles_rdkit_valid` | RDKit round-trip check on the *input* SMILES. Defends against fragrec's `frag(...)` / `recomb(...)` virtual species whose SMILES strings are syntactically OK but violate valence: a single bad fragment crashes FlowER's `Chem.AddHs(NoneType)` and (because `beam_predict.py` has no per-row recovery) loses the entire batch. |

Routing is exposed as `route_combo(combo) → {'flower' | 'fragrec'}` and
is the canonical place to extend the policy.

## Path-cost contributions

`src/thermo.py::reaction_cost` is

```
cost =   softplus(ΔG, scale)
       − λ · log(max(P_FlowER, p_floor))
       + tier_bias[ tier ]                   # only if --use-tier-bias
```

with three orthogonal knobs.

### `λ` (FlowER prior weight)

Default `λ = 0` reproduces the paper's original cost exactly.
`P_FlowER ∈ [0, 1]` is `count / sample_size` from FlowER's beam search.
Reactions FlowER did not produce get `p_floor = 0.01` to keep
`−log P_FlowER` finite for non-organic Li-coordination steps.

### `--use-tier-bias`

Each reaction is tagged with a `CandidateTier` based on which backend
produced it and how confident FlowER is:

| Tier | Source | Bias (default) |
| ---- | ------ | -------------- |
| `FLOWER_HIGH` | FlowER prior ≥ 0.5 | −0.5 |
| `FLOWER_FRAGREC` | both backends agree | −0.2 |
| `FRAGREC_ORGANIC` | fragrec only, organic combo | +0.3 |
| `FRAGREC_INORGANIC` | fragrec only, contains Li / charged / radical | 0.0 (neutral) |

The bias only fires when `--use-tier-bias` is set. Calibration is open
research — the defaults above were chosen before real FlowER data was
available and are likely too aggressive; expect to tune them once you
have run the demo with `--flower-as-generator` on a representative
substrate set.

### `p_floor`

Only relevant when `λ > 0`. Default `0.01`. Pinning above zero stops
`−log` from dominating the cost of any reaction FlowER didn't get to
opine on.

## CLI

```bash
# Default: paper-only, fragrec only, no FlowER calls
python -m src.run_demo

# Re-ranker mode (legacy v1): FlowER attaches priors to existing rxns
python -m src.run_demo --backend both --lam 0.5

# Generator mode (v2, opt-in): FlowER products extend the pool
python -m src.run_demo \
    --backend both \
    --flower-as-generator \
    --max-pool-iterations 1 \
    --flower-prob-threshold 0.05 \
    --combo-sizes 1,2 \
    --use-tier-bias \
    --lam 0.5
```

Generator-mode flags (all optional):

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--max-pool-iterations` | 1 | How many expansion rounds. Each round re-enumerates combos over the *current* pool, so this can explode quickly. |
| `--flower-prob-threshold` | 0.05 | Discard FlowER predictions with `P_FlowER` below this. Also applied by `is_acceptable_flower_product`. |
| `--combo-sizes` | `1,2` | Comma-separated. Size-3 supported but slow on CPU (3-body inference takes 2–5 min/combo). |
| `--use-tier-bias` | off | Apply per-tier additive bias above. |

## Cache

`src/flower_backend.py` writes a JSON file to `cache/flower/` for every
reactant set it predicts. The filename is a SHA-1 of
`(canonical_dot_smiles, model_name, sample_size)`, so the cache is
deterministic and **portable** between machines. Each file is a JSON
list of `[product_dot_smiles, count]` pairs:

```json
[["CO.OC=O", 25], ["O=C(O)OCC", 12]]
```

`count / sample_size` is the probability assigned to each predicted
product set. The order is FlowER's beam-search ranking, top-`nbest`
only (default 3). On a cache hit, no subprocess is spawned, so a typical
"shipped cache" workflow is:

1. Run FlowER once on a fast machine; populate `cache/flower/`.
2. `rsync -av host:~/.../cache/flower/ ./cache/flower/`.
3. On the slow / GPU-less machine, `python -m src.run_demo --backend both`
   resolves every prediction from cache. `is_available()` is allowed to
   return `False` as long as no cache miss occurs; only a miss raises
   `FlowERUnavailable`.

## Subprocess robustness

* FlowER is invoked in **chunks of `chunk_size=50`** (configurable via
  `FLOWER_CHUNK_SIZE`). A chunk that times out or crashes is logged and
  skipped; partial cache writes from earlier chunks are preserved. This
  matters because `beam_predict.py` has no per-row error recovery — a
  single bad SMILES forfeits the whole batch.
* The chunk timeout is `timeout_s = 1800` (30 min) by default. On CPU,
  one chunk of 50 organic 2-body combos takes ~5–15 min.
* `_smiles_rdkit_valid` (Layer 3 above) intercepts the most common
  source of bad SMILES: fragrec's dangling-bond virtual species.

## Calibration probe

`src/flower_probe.py` runs FlowER on 10 LEDC-relevant combos
(EC + H₂O, EC + EC, VC + H₂O, EC + MeOH, EC + MeNH₂, DMC + H₂O, …) and
prints both the formula-level and SMILES-level top predictions. It is
the recommended sanity check before trusting `_TIER_BIAS` values:

```bash
python -m src.flower_probe
```

On a representative run, FlowER correctly predicted "no reaction" for 9
out of 10 control combos and recovered the textbook tetrahedral
intermediate for EC + MeNH₂ (aminolysis). It also produced one chemical
hallucination (VC → furanone-like aromatic) — a useful reminder that
FlowER is not infallible on substrates outside its training distribution.

## Troubleshooting

* **"FlowER repo or checkpoint not configured."** — set
  `FLOWER_REPO_PATH` and `FLOWER_MODEL_PATH`. Both `FLOWER_<NAME>` and
  bare `<NAME>` are recognised; `FLOWER_*` is preferred to avoid
  colliding with FlowER's own `MODEL_PATH` etc. when the upstream env
  is also active.
* **Subprocess returns non-zero** — `FlowERUnavailable` carries the
  last 2 KB of stderr. Common cause: wrong `EMB_DIM` for the chosen
  checkpoint. The default 256 matches `model.2880000_95.pt`; the
  smaller released models use 128.
* **`AddHs(NoneType)` Boost.Python ArgumentError** — Layer-3 valence
  guard should have caught this; if you still see it, the SMILES
  reaching FlowER bypassed `expand()`. File an issue.
* **Cache hit rate < 100 % on a downstream machine** — your seeds or
  `combo-sizes` differ between hosts. Either align the parameters or
  pre-warm the cache by running `--flower-as-generator` once with the
  union of all combo-sizes you expect to need.

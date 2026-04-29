# Reaction Network Reproduction — JACS 2021, 143, 13245

Educational mini-reproduction of:

> Xie, Spotte-Smith, Wen, Patel, Blau, Persson.
> *Data-Driven Prediction of Formation Mechanisms of Lithium Ethylene
> Monocarbonate with an Automated Reaction Network.*
> **J. Am. Chem. Soc. 2021, 143, 13245–13258.**
> https://doi.org/10.1021/jacs.1c05807

with optional **FlowER** mechanistic prior (Joung & Fong et al.,
*Nature* **645**, 115 (2025); https://doi.org/10.1038/s41586-025-09426-9).

## Scope

Reproduces the **methodological pipeline** of the paper at a teaching
level — runnable on a laptop, no DFT, no proprietary tooling. Target is
**LEDC** (lithium ethylene dicarbonate), the canonical SEI organic
component, formed from `{EC, Li⁺, H₂O, e⁻}`.

| Stage | Paper | This repo |
|---|---|---|
| Molecule = labelled graph | §2.1 | `src/molecule.py` (NetworkX) |
| n-step fragmentation | §2.1 | `src/fragrec.py::fragment` |
| One-step recombination | §2.1 | `src/fragrec.py::recombine` |
| Graph isomorphism dedup | §2.1 | `nx.is_isomorphic` |
| BonDNet ΔG_recomb filter | §2.1 (3) | mock predictor (`src/bde_model.py`) |
| Concerted reaction enumeration (CD ≤ k) | §2.2 | `src/network.py::enumerate_reactions` |
| Redox at U=0 vs Li/Li⁺ (μ_e = −1.40 eV) | §2.2 | `src/thermo.py` |
| softplus(ΔG) cost + shortest path | §2.2 | `src/pathfind.py` (Dijkstra + Yen K) |
| DFT geometry / TS refinement (ωB97X-V) | §2.3 | **NOT included** — out of scope |
| FlowER mechanistic prior | (extension) | `src/flower_backend.py` (subprocess + cache) |

Mock energies are used so the pipeline runs end-to-end. Real DFT free
energies from the published Materials Project / lithium-ion electrolyte
dataset (Spotte-Smith et al., *Sci. Data* 2021) can be plugged in by
replacing the heuristic in `src/run_demo.py::mock_free_energy`.

## Quick start

### Option A — conda (recommended, reproduces the dev env)

```bash
git clone https://github.com/<your-handle>/ReactionNetwork_jacs2021.git
cd ReactionNetwork_jacs2021
conda env create -f environment.yml
conda activate jacs2021
python -m src.run_demo
```

### Option B — pip

```bash
git clone https://github.com/<your-handle>/ReactionNetwork_jacs2021.git
cd ReactionNetwork_jacs2021
pip install -r requirements.txt
python -m src.run_demo
```

Expect ~30 s on a laptop. Output: top-K shortest reaction pathways from
`{EC, Li⁺, H₂O, e⁻}` to LEDC, with per-step ΔG, written to
`results/ledc_paths.json`.

### CLI flags

```bash
python -m src.run_demo --backend fragrec               # default, paper-only pipeline
python -m src.run_demo --backend both --lam 0.5        # add FlowER prior
python -m src.run_demo --max-bond-changes 4 --n-paths 10
```

`--backend both` requires either a working FlowER install (env vars
`FLOWER_REPO_PATH`, `FLOWER_MODEL_PATH`) **or** a populated
`cache/flower/` directory. See `docs/FLOWER_INTEGRATION.md`.

## Layout

```
ReactionNetwork_jacs2021/
├── README.md
├── environment.yml          # conda env (Python 3.11 + RDKit + NetworkX)
├── requirements.txt         # pip alternative
├── src/
│   ├── molecule.py          # MoleculeGraph: labelled graph + canonical hash
│   ├── fragrec.py           # fragmentation + recombination (§2.1)
│   ├── bde_model.py         # mock BDE / recombination ΔG (BonDNet stand-in)
│   ├── thermo.py            # free energies, redox, hybrid cost
│   ├── network.py           # species pool + concerted reaction enumeration
│   ├── pathfind.py          # softplus cost, Dijkstra, Yen's K-shortest paths
│   ├── smiles_bridge.py     # SMILES <-> MoleculeGraph (RDKit)
│   ├── flower_backend.py    # FlowER subprocess adapter + JSON cache
│   └── run_demo.py          # end-to-end CLI
├── data/
│   └── seed_species.py      # EC, Li⁺, H₂O, LEDC target
├── tests/                   # pytest suite (22 tests, no GPU needed)
├── docs/
│   └── FLOWER_INTEGRATION.md
├── notebooks/
│   └── walkthrough.ipynb
└── results/                 # generated paths & figures
```

## Testing

```bash
python -m pytest tests/ -q
```

22 tests; all run on the conda env without a GPU or FlowER checkpoint
(cache-only paths are exercised).

## FlowER integration (optional)

FlowER handles **organic** substeps; fragrec retains everything Li- /
charge- / radical-bearing. The hybrid path cost is

```
cost = softplus(ΔG, scale) − λ · log P_FlowER
```

with `λ = 0` recovering the paper's original cost exactly. See
`docs/FLOWER_INTEGRATION.md` for GPU-host setup and the laptop cache
workflow (run FlowER once on a GPU box, copy `cache/flower/` to your
laptop, run with `--backend both` offline).

## Key references for the production pipeline

- HiPRGen (production reaction-network builder): https://github.com/BlauGroup/HiPRGen
- BonDNet (graph-NN BDE predictor): https://github.com/mjwen/bondnet
- mrnet / pymatgen: https://github.com/materialsproject
- FlowER: https://github.com/FongMunHong/FlowER

## Caveats

This is a **conceptual** reproduction. The real paper's quantitative
conclusions (e.g. "LEDC hydrolysis barrier 1.48 eV") cannot be recovered
without running the DFT stack on QChem with ωB97X-V/def2-TZVPPD/SMD.
This code is intended to make the *algorithms* legible, not to replace
the *calculations*.

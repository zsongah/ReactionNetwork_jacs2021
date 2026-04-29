"""FlowER backend adapter.

This module lets the species-pool builder optionally use FlowER
(Joung & Fong et al., *Nature* **645**, 115 (2025);
https://github.com/FongMunHong/FlowER) to expand reactant combinations
into mechanistically-plausible products.

Design constraints
------------------
* No FlowER imports at module-load time. The adapter spawns FlowER as a
  *subprocess*, so this project remains importable on machines without
  PyTorch / a CUDA stack.
* All filesystem I/O lives under a dedicated cache directory. A second
  invocation with the same input is served from disk.
* If FlowER cannot be located or its execution fails, a
  ``FlowERUnavailable`` exception is raised; callers fall back to
  ``fragrec``.
* The adapter is *organic-only*. Reactant groups containing Li, free
  electrons, or unusual oxidation states are filtered out before
  invocation (those go through fragrec). This matches the user's
  decision to let fragrec handle electrochemistry while FlowER handles
  the organic interior.

Subprocess protocol (mirrors run_FlowER_large_*Data.sh + scripts/search.sh)
--------------------------------------------------------------------------
Input file (``$TEST_FILE``)
    One reactant set per line, ``A.B.C`` SMILES (or ``A.B>>P1|P2`` to
    seed the targeted search).
Output
    Pickles at ``$RESULT_PATH/result_chunk_{i}_s{seed}.pickle`` whose
    contents are ``[(graph, root, (reactant, products), check), ...]``.
    Each ``graph`` is a ``networkx.DiGraph`` whose nodes are SMILES
    and whose edges carry ``rank`` and ``count`` attributes.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .molecule import MoleculeGraph
from .smiles_bridge import (
    BridgeError,
    dot_smiles_to_mols,
    mol_to_smiles,
    mols_to_dot_smiles,
)


class FlowERUnavailable(RuntimeError):
    """Raised when the FlowER subprocess cannot be invoked.

    Callers typically catch this and fall back to ``fragrec``.
    """


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class FlowERConfig:
    """Locations and hyper-parameters needed to launch FlowER.

    Defaults follow ``run_FlowER_large_newData.sh``. Override any of
    these per machine; the fields all map directly to FlowER's expected
    environment variables.
    """

    repo_path: Optional[Path] = None       # FlowER git checkout root
    model_path: Optional[Path] = None      # directory containing the .pt
    model_name: str = "model.2880000_95.pt"
    data_name: str = "flower_new_dataset"
    exp_name: str = "best_large_hyperparam"
    emb_dim: int = 256
    rbf_high: float = 12.0
    rbf_gap: float = 0.1
    sigma: float = 0.15
    sample_size: int = 50
    beam_size: int = 5
    nbest: int = 3
    max_depth: int = 6
    chunk_size: int = 50
    test_batch_size: int = 4096
    python_executable: str = "python"      # interpreter inside FlowER's env
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "cache" / "flower"
    )
    timeout_s: int = 1800

    @classmethod
    def from_env(cls) -> "FlowERConfig":
        """Construct from environment variables of the same names
        (uppercase). Everything missing keeps its default."""
        kwargs: dict = {}
        for f in ("repo_path", "model_path", "cache_dir"):
            v = os.environ.get(f.upper())
            if v:
                kwargs[f] = Path(v)
        for f in (
            "model_name", "data_name", "exp_name", "python_executable",
        ):
            v = os.environ.get(f.upper())
            if v:
                kwargs[f] = v
        for f, cast in (
            ("emb_dim", int), ("rbf_high", float), ("rbf_gap", float),
            ("sigma", float), ("sample_size", int), ("beam_size", int),
            ("nbest", int), ("max_depth", int), ("chunk_size", int),
            ("test_batch_size", int), ("timeout_s", int),
        ):
            v = os.environ.get(f.upper())
            if v:
                kwargs[f] = cast(v)
        return cls(**kwargs)


# ----------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------
@dataclass
class FlowERPrediction:
    """One predicted product set from a single reactant combo."""

    reactants: tuple[MoleculeGraph, ...]
    products: tuple[MoleculeGraph, ...]
    probability: float          # count / sample_size, in [0, 1]


class FlowERBackend:
    """Subprocess-backed FlowER wrapper with on-disk caching."""

    def __init__(self, config: Optional[FlowERConfig] = None):
        self.cfg = config or FlowERConfig.from_env()
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Cheap pre-flight check. Does NOT actually invoke FlowER."""
        if self.cfg.repo_path is None or not self.cfg.repo_path.exists():
            return False
        if not (self.cfg.repo_path / "beam_predict.py").exists():
            return False
        if self.cfg.model_path is None or not self.cfg.model_path.exists():
            return False
        return True

    def expand(
        self,
        reactant_groups: Iterable[tuple[MoleculeGraph, ...]],
    ) -> list[FlowERPrediction]:
        """Run FlowER on each reactant tuple.

        The list is filtered to *organic-only* reactant groups before
        invocation (see :func:`is_organic_combo`). Skipped groups produce
        no predictions and the caller is expected to handle them via
        fragrec.

        Cache key is the canonical multiset of reactant SMILES plus the
        model checkpoint name; a hit avoids the subprocess entirely.
        """
        # Step 1: render & filter
        rendered: list[tuple[tuple[MoleculeGraph, ...], str]] = []
        for combo in reactant_groups:
            if not is_organic_combo(combo):
                continue
            try:
                smi = mols_to_dot_smiles(list(combo))
            except BridgeError:
                continue
            rendered.append((combo, smi))

        if not rendered:
            return []

        # Step 2: cache check
        results: list[FlowERPrediction] = []
        misses: list[tuple[tuple[MoleculeGraph, ...], str]] = []
        for combo, smi in rendered:
            cached = self._cache_load(smi)
            if cached is not None:
                results.extend(self._materialize(combo, cached))
            else:
                misses.append((combo, smi))

        # Step 3: subprocess for misses
        if misses:
            # Only require a configured FlowER install when we actually
            # need to run the model. Cache-only execution is supported on
            # machines without GPUs (typical dev workflow: run FlowER
            # once on a GPU host, ship cache/ to the laptop).
            if not self.is_available():
                raise FlowERUnavailable(
                    "FlowER repo or checkpoint not configured. "
                    "Set FLOWER_REPO_PATH and FLOWER_MODEL_PATH or pass "
                    "FlowERConfig explicitly. "
                    f"({len(misses)} reactant set(s) missed cache.)"
                )
            invoked = self._run_subprocess([smi for _, smi in misses])
            for (combo, smi), per_reactant in zip(misses, invoked):
                self._cache_store(smi, per_reactant)
                results.extend(self._materialize(combo, per_reactant))

        return results

    # ------------------------------------------------------------------
    # Cache layer
    # ------------------------------------------------------------------
    def _cache_key(self, reactant_smiles: str) -> str:
        canonical = ".".join(sorted(reactant_smiles.split(".")))
        payload = f"{canonical}|{self.cfg.model_name}|{self.cfg.sample_size}"
        return hashlib.sha1(payload.encode()).hexdigest()[:24]

    def _cache_load(
        self, reactant_smiles: str,
    ) -> Optional[list[tuple[str, int]]]:
        path = self.cfg.cache_dir / f"{self._cache_key(reactant_smiles)}.json"
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    def _cache_store(
        self, reactant_smiles: str, products: list[tuple[str, int]],
    ) -> None:
        path = self.cfg.cache_dir / f"{self._cache_key(reactant_smiles)}.json"
        with path.open("w") as f:
            json.dump(products, f)

    # ------------------------------------------------------------------
    # Result materialization
    # ------------------------------------------------------------------
    def _materialize(
        self,
        reactants: tuple[MoleculeGraph, ...],
        predictions: list[tuple[str, int]],
    ) -> list[FlowERPrediction]:
        out: list[FlowERPrediction] = []
        for prod_smi, count in predictions:
            try:
                prods = tuple(dot_smiles_to_mols(prod_smi))
            except BridgeError:
                continue
            p = count / max(self.cfg.sample_size, 1)
            out.append(FlowERPrediction(
                reactants=tuple(reactants),
                products=prods,
                probability=float(p),
            ))
        return out

    # ------------------------------------------------------------------
    # Subprocess invocation
    # ------------------------------------------------------------------
    def _run_subprocess(
        self, reactant_lines: list[str],
    ) -> list[list[tuple[str, int]]]:
        """Spawn FlowER's ``beam_predict.py`` and parse its pickles.

        Returns a list aligned with ``reactant_lines``; each element is a
        ``[(product_smiles, count), ...]`` list.
        """
        repo = self.cfg.repo_path
        # Stage input/output dirs inside cache_dir so we can reproduce.
        with tempfile.TemporaryDirectory(prefix="flower_run_", dir=self.cfg.cache_dir) as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data" / self.cfg.data_name
            data_dir.mkdir(parents=True)
            beam_txt = data_dir / "beam.txt"
            beam_txt.write_text("\n".join(reactant_lines) + "\n")

            result_dir = tmp_path / "results" / self.cfg.data_name / self.cfg.exp_name
            result_dir.mkdir(parents=True)

            env = self._build_env(beam_txt, result_dir)

            # We invoke beam_predict.py directly rather than the shell
            # wrapper, since the wrapper toggles training/eval as well.
            cmd = [self.cfg.python_executable, "beam_predict.py"]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_s,
                    check=False,
                )
            except FileNotFoundError as e:
                raise FlowERUnavailable(f"python interpreter not found: {e}")
            except subprocess.TimeoutExpired as e:
                raise FlowERUnavailable(
                    f"FlowER timed out after {self.cfg.timeout_s}s"
                ) from e

            if proc.returncode != 0:
                raise FlowERUnavailable(
                    f"FlowER subprocess exited with code {proc.returncode}.\n"
                    f"stderr (last 2 KB):\n{proc.stderr[-2048:]}"
                )

            return self._parse_results(result_dir, reactant_lines)

    def _build_env(self, beam_txt: Path, result_dir: Path) -> dict:
        env = os.environ.copy()
        env.update({
            "DATA_NAME": self.cfg.data_name,
            "EXP_NAME": self.cfg.exp_name,
            "MODEL_NAME": self.cfg.model_name,
            "MODEL_PATH": str(self.cfg.model_path) + os.sep,
            "EMB_DIM": str(self.cfg.emb_dim),
            "RBF_HIGH": str(self.cfg.rbf_high),
            "RBF_GAP": str(self.cfg.rbf_gap),
            "SIGMA": str(self.cfg.sigma),
            "TEST_BATCH_SIZE": str(self.cfg.test_batch_size),
            "TEST_FILE": str(beam_txt),
            "RESULT_PATH": str(result_dir) + os.sep,
            "SCALE": "1",
            "NUM_WORKERS": "0",
            # multi-GPU vars set so the single-process path is taken
            "NUM_GPUS_PER_NODE": "1",
            "NUM_NODES": "1",
            "NODE_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "1235",
        })
        return env

    def _parse_results(
        self, result_dir: Path, reactant_lines: list[str],
    ) -> list[list[tuple[str, int]]]:
        """Read FlowER pickles and project each DiGraph to (product, count).

        The DiGraph's *root* node is the input reactant SMILES. Each direct
        successor edge ``(root -> product)`` carries a ``count`` attribute
        from FlowER's beam search. We aggregate over all such direct
        children — the immediate one-step expansion — because that matches
        the granularity of fragrec.
        """
        # Load every pickle; build a map from canonical reactant smi to
        # aggregated product counts.
        canonical_input = [self._canon_dotsmiles(s) for s in reactant_lines]
        per_reactant: dict[str, dict[str, int]] = {s: {} for s in canonical_input}

        for pkl in sorted(result_dir.glob("result_chunk_*.pickle")):
            with pkl.open("rb") as f:
                payload = pickle.load(f)
            for graph, root, (orig_reactant, _wanted), _check in payload:
                canon_root = self._canon_dotsmiles(orig_reactant)
                if canon_root not in per_reactant:
                    continue
                bucket = per_reactant[canon_root]
                # one-step children of the root
                if root not in graph:
                    continue
                for child in graph.successors(root):
                    edge = graph.get_edge_data(root, child) or {}
                    count = int(edge.get("count", 0))
                    if count <= 0:
                        continue
                    canon_child = self._canon_dotsmiles(child)
                    bucket[canon_child] = bucket.get(canon_child, 0) + count

        # Materialize in input order, truncated to nbest.
        out: list[list[tuple[str, int]]] = []
        for s in canonical_input:
            items = sorted(per_reactant[s].items(), key=lambda kv: -kv[1])
            out.append(items[: self.cfg.nbest])
        return out

    @staticmethod
    def _canon_dotsmiles(smi: str) -> str:
        from rdkit import Chem
        parts = []
        for f in smi.split("."):
            m = Chem.MolFromSmiles(f, sanitize=False)
            if m is None:
                parts.append(f)
            else:
                parts.append(Chem.MolToSmiles(m, canonical=True))
        return ".".join(sorted(parts))


# ----------------------------------------------------------------------
# Filtering: what counts as "organic"?
# ----------------------------------------------------------------------
def is_organic_combo(combo: Iterable[MoleculeGraph]) -> bool:
    """FlowER training data does not include Li / explicit free electrons.

    We restrict it to neutral, closed-shell organic combinations. Net
    charges are allowed only if they balance (e.g., R-O- + H+) but the
    safer default in this project is to route any charged combo through
    fragrec, since the paper's electrochemistry is the bigger weakness
    of FlowER's training distribution.
    """
    total_charge = 0
    total_spin = 0
    for m in combo:
        total_charge += m.charge
        total_spin += m.spin
        for _, d in m.graph.nodes(data=True):
            if d["element"] in {"Li", "Na", "K", "Mg", "Ca"}:
                return False
    return total_charge == 0 and total_spin == 0

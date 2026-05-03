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
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
import time
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


log = logging.getLogger(__name__)


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
        """Construct from environment variables.

        Two prefixes are supported, in order:
          1. ``FLOWER_<NAME>`` — preferred, namespaced.
          2. ``<NAME>`` — raw uppercase (kept for back-compat with the
             FlowER upstream ``run_FlowER_*.sh`` convention).

        Anything missing keeps its dataclass default.
        """
        def _lookup(name: str) -> Optional[str]:
            return (
                os.environ.get(f"FLOWER_{name.upper()}")
                or os.environ.get(name.upper())
            )

        kwargs: dict = {}
        for f in ("repo_path", "model_path", "cache_dir"):
            v = _lookup(f)
            if v:
                kwargs[f] = Path(v)
        for f in (
            "model_name", "data_name", "exp_name", "python_executable",
        ):
            v = _lookup(f)
            if v:
                kwargs[f] = v
        for f, cast in (
            ("emb_dim", int), ("rbf_high", float), ("rbf_gap", float),
            ("sigma", float), ("sample_size", int), ("beam_size", int),
            ("nbest", int), ("max_depth", int), ("chunk_size", int),
            ("test_batch_size", int), ("timeout_s", int),
        ):
            v = _lookup(f)
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


def _smiles_rdkit_valid(smi: str) -> bool:
    """Cheap RDKit round-trip check.

    Returns True only if every '.'-separated fragment parses *and*
    survives ``Chem.AddHs(sanitize=True)``. We mirror what FlowER's
    ``beam_predict.reactant_process`` does so we can intercept inputs
    that would otherwise crash the FlowER subprocess on the first
    fragment and lose the entire batch (Boost.Python ArgumentError on
    NoneType, no per-row recovery available without vendor patch).
    """
    Chem = _get_rdkit_chem()
    if Chem is None:
        return True  # if RDKit absent, trust upstream and don't filter
    for frag in smi.split("."):
        if not frag:
            return False
        m = Chem.MolFromSmiles(frag)
        if m is None:
            return False
        try:
            Chem.AddHs(m, explicitOnly=False)
        except Exception:
            return False
    return True


_RDKIT_CHEM = None
_RDKIT_INIT_DONE = False


def _get_rdkit_chem():
    """Lazy-import RDKit once and silence its logger; return Chem or None."""
    global _RDKIT_CHEM, _RDKIT_INIT_DONE
    if _RDKIT_INIT_DONE:
        return _RDKIT_CHEM
    _RDKIT_INIT_DONE = True
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        _RDKIT_CHEM = Chem
    except Exception:
        _RDKIT_CHEM = None
    return _RDKIT_CHEM


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
        *,
        min_probability: float = 0.0,
    ) -> list[FlowERPrediction]:
        """Run FlowER on each reactant tuple.

        The list is filtered to *FlowER-routable* reactant groups before
        invocation (see :func:`route_combo`). Skipped groups produce no
        predictions and the caller is expected to handle them via fragrec.

        Cache key is the canonical multiset of reactant SMILES plus the
        model checkpoint name; a hit avoids the subprocess entirely.

        Parameters
        ----------
        min_probability
            If > 0, predictions whose materialised probability falls below
            this threshold are discarded. The same threshold is used by
            :func:`is_acceptable_flower_product` if you wire it through.
        """
        # Step 1: render & filter
        rendered: list[tuple[tuple[MoleculeGraph, ...], str]] = []
        for combo in reactant_groups:
            if route_combo(combo) != "flower":
                continue
            try:
                smi = mols_to_dot_smiles(list(combo))
            except BridgeError:
                continue
            # Layer-3 guard: defend against fragrec virtual species
            # (frag/recomb nodes) whose SMILES strings are syntactically
            # OK but violate valence rules. FlowER's beam_predict.py calls
            # AddHs() on the parsed mol and crashes the whole batch on a
            # single None — we cannot recover from a partial-batch failure
            # without patching vendor code. Round-trip through RDKit
            # ourselves; skip combos RDKit itself rejects.
            if not _smiles_rdkit_valid(smi):
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
                results.extend(self._materialize(combo, cached, min_probability))
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
            # Chunk the misses so a single slow / failing batch does not
            # forfeit all FlowER work for this expansion round. Each chunk
            # gets its own subprocess (fresh torch import overhead, but
            # that's ~10–15 s vs many minutes saved on partial failures).
            chunk = max(1, int(self.cfg.chunk_size))
            n_chunks = (len(misses) + chunk - 1) // chunk
            t_start = time.time()
            for start in range(0, len(misses), chunk):
                batch = misses[start : start + chunk]
                idx = start // chunk + 1
                log.info(
                    "FlowER chunk %d/%d: %d combos (elapsed %.0f s)",
                    idx, n_chunks, len(batch), time.time() - t_start,
                )
                try:
                    invoked = self._run_subprocess([smi for _, smi in batch])
                except FlowERUnavailable as e:
                    log.warning(
                        "FlowER chunk %d/%d failed (%d combos): %s. "
                        "Skipping this chunk; cached partial progress kept.",
                        idx, n_chunks, len(batch), e,
                    )
                    continue
                for (combo, smi), per_reactant in zip(batch, invoked):
                    self._cache_store(smi, per_reactant)
                    results.extend(self._materialize(combo, per_reactant, min_probability))

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
        # Atomic write: a SIGINT during the write must not leave a
        # half-written file that the next run trusts as a cache hit.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(products, f)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Result materialization
    # ------------------------------------------------------------------
    def _materialize(
        self,
        reactants: tuple[MoleculeGraph, ...],
        predictions: list[tuple[str, int]],
        min_probability: float = 0.0,
    ) -> list[FlowERPrediction]:
        out: list[FlowERPrediction] = []
        for prod_smi, count in predictions:
            try:
                prods = tuple(dot_smiles_to_mols(prod_smi))
            except BridgeError:
                continue
            p = count / max(self.cfg.sample_size, 1)
            if min_probability > 0 and p < min_probability:
                continue
            # Layer 2: post-output sanity / conservation check.
            if not is_acceptable_flower_product(
                prods, reactants, p,
                min_probability=max(min_probability, 1e-6),
            ):
                continue
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
# Filtering layers
# ----------------------------------------------------------------------
# These are the gates that decide *whether* a candidate combination of
# species should be sent to FlowER. They implement the policy laid out in
# docs/FLOWER_INTEGRATION.md:
#
#   Layer 0 (per-species)   is_flower_compatible
#       Reject species containing metals, multi-charges, or atoms outside
#       FlowER's training distribution.
#
#   Layer 1 (per-combo)     is_reactive_combo
#       Reject combos with no plausible reactive site, oversized BE matrix,
#       or implausible mixtures.
#
#   Layer 2 (post-output)   is_acceptable_flower_product
#       Reject FlowER outputs that violate conservation, contain atoms
#       outside the input set, or look like hallucinations.
#
#   Routing                 route_combo
#       Decides "flower" vs "fragrec" for a given combo.

_METALS = {"Li", "Na", "K", "Mg", "Ca", "Fe", "Co", "Ni", "Cu", "Zn"}
_FLOWER_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def is_flower_compatible(species: MoleculeGraph) -> bool:
    """Layer 0: can FlowER's BE-matrix architecture even represent this?

    Hard architectural / training-distribution gates:
      * No metals (Li OOD, transition metals 100% OOD).
      * Net charge zero. Charged species (especially radical anions like
        EC•⁻) are very rare in USPTO+RmechDB+PmechDB.
      * Heavy-atom count <= 30 (FlowER training tops out around there).
      * All elements drawn from the FlowER element vocabulary.

    Neutral radicals (spin > 0, charge 0) are *allowed* — RmechDB
    contributes ~5K such steps to FlowER's training set.
    """
    if abs(species.charge) > 0:
        return False
    heavy = 0
    for _, d in species.graph.nodes(data=True):
        el = d["element"]
        if el in _METALS:
            return False
        if el not in _FLOWER_ELEMENTS:
            return False
        if el != "H":
            heavy += 1
    if heavy > 30:
        return False
    return True


def has_reactive_site(species: MoleculeGraph) -> bool:
    """Cheap heuristic: does this molecule have *anything* that could react?

    Used by ``is_reactive_combo`` to skip pairings of inert saturated
    closed-shell molecules where FlowER would only hallucinate.

    A species is considered reactive if any of these is true:
      * it carries an unpaired electron (radical),
      * it has at least one multi-bond (double / triple, ``order >= 2``),
      * it contains O, N, S, P, or a halogen (lone-pair donors / acceptors
        and weak bonds).
    """
    if species.spin > 0:
        return True
    for _, _, edata in species.graph.edges(data=True):
        if isinstance(edata, dict) and edata.get("order", 1) >= 2:
            return True
    for _, d in species.graph.nodes(data=True):
        if d["element"] in {"O", "N", "S", "P", "F", "Cl", "Br", "I"}:
            return True
    return False


def is_reactive_combo(
    combo: Iterable[MoleculeGraph],
    *,
    max_total_heavy_atoms: int = 25,
    max_size: int = 3,
) -> bool:
    """Layer 1: is this combination plausibly reactive?

    Rules:
      1. Each species must pass ``is_flower_compatible``.
      2. Combo size must lie in [1, max_size]. FlowER's training data is
         dominated by 2-element combos with a handful of 3-element ones
         (radical + two substrates). 4+ is OOD.
      3. Total heavy atoms <= max_total_heavy_atoms; otherwise the BE
         matrix is too large for FlowER's typical inference window.
      4. At least one species must have a reactive site, except for size=1
         where the molecule itself must have a reactive site (otherwise
         a saturated closed-shell molecule has nothing to rearrange).
      5. No more than 2 copies of the same species (3+ identical fragments
         is essentially never seen in training).
    """
    combo_list = list(combo)
    n = len(combo_list)
    if n == 0 or n > max_size:
        return False

    if not all(is_flower_compatible(m) for m in combo_list):
        return False

    total_heavy = 0
    for m in combo_list:
        for _, d in m.graph.nodes(data=True):
            if d["element"] != "H":
                total_heavy += 1
    if total_heavy > max_total_heavy_atoms:
        return False

    if not any(has_reactive_site(m) for m in combo_list):
        return False

    # Detect 3+ identical species via canonical hash.
    seen: dict[str, int] = {}
    for m in combo_list:
        h = m.canonical_hash()
        seen[h] = seen.get(h, 0) + 1
    if max(seen.values()) >= 3:
        return False

    return True


def route_combo(combo: Iterable[MoleculeGraph]) -> str:
    """Decide which backend handles this reactant combination.

    Returns
    -------
    "flower"
        Pure-organic, neutral, with at least one reactive site. FlowER
        will be queried; products feed back into the species pool.
    "fragrec"
        Anything containing Li/metal/charge, plus combos rejected by the
        reactive-combo filter. fragrec's combinatorial enumeration covers
        these (electrochemistry, Li coordination, radical anions, ...).
    """
    combo_list = list(combo)
    if not combo_list:
        return "fragrec"
    # Quick reject: any metal anywhere.
    for m in combo_list:
        for _, d in m.graph.nodes(data=True):
            if d["element"] in _METALS:
                return "fragrec"
    # Net-charge check: combos that as a whole carry charge are routed to
    # fragrec because radical-anion / cation chemistry is FlowER-OOD.
    if sum(m.charge for m in combo_list) != 0:
        return "fragrec"
    if any(m.charge != 0 for m in combo_list):
        return "fragrec"
    if not is_reactive_combo(combo_list):
        return "fragrec"
    return "flower"


def is_acceptable_flower_product(
    products: Iterable[MoleculeGraph],
    reactants: Iterable[MoleculeGraph],
    probability: float,
    *,
    min_probability: float = 0.05,
) -> bool:
    """Layer 2: should we accept a FlowER prediction into the pool?

    FlowER occasionally hallucinates products that violate basic
    conservation — typically when its beam search exits early on a
    low-confidence node. We screen them out before they pollute downstream
    enumeration.

    Checks:
      * probability >= min_probability,
      * no metals introduced (FlowER should not invent Li),
      * heavy-atom count conservation: |Σ heavy(prod) − Σ heavy(react)| <= 1
        (allow a tiny slack for cases where an explicit H is dropped),
      * net charge conservation,
      * no implausible product growth (Σ heavy(prod) > 2 × Σ heavy(react)).
    """
    if probability < min_probability:
        return False
    prod_list = list(products)
    react_list = list(reactants)
    if not prod_list or not react_list:
        return False

    def heavy(mols: list[MoleculeGraph]) -> int:
        return sum(
            1 for m in mols for _, d in m.graph.nodes(data=True)
            if d["element"] != "H"
        )

    # No metals introduced (only check products; reactants were already
    # vetted by is_flower_compatible).
    for m in prod_list:
        for _, d in m.graph.nodes(data=True):
            if d["element"] in _METALS:
                return False

    # Net charge conservation.
    if sum(m.charge for m in prod_list) != sum(m.charge for m in react_list):
        return False

    h_react = heavy(react_list)
    h_prod = heavy(prod_list)
    if h_react == 0:
        return False
    if abs(h_prod - h_react) > 1:
        return False
    if h_prod > 2 * h_react:
        return False

    return True


# ----------------------------------------------------------------------
# Backward-compatible alias retained for callers/tests written against
# the original strict (closed-shell, neutral, metal-free) gate.
# ----------------------------------------------------------------------
def is_organic_combo(combo: Iterable[MoleculeGraph]) -> bool:
    """Strictest legacy gate: neutral, closed-shell, metal-free.

    Equivalent to ``is_reactive_combo`` *plus* ``total_spin == 0``.
    Kept for backwards compatibility with tests that pre-date the
    layered filter design.
    """
    combo_list = list(combo)
    if not is_reactive_combo(combo_list):
        return False
    return sum(m.spin for m in combo_list) == 0

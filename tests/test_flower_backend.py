"""Tests for src.flower_backend that do NOT require a working FlowER install.

We exercise three things:

1. ``is_organic_combo`` correctly rejects Li-containing, charged, and
   radical reactant sets (those go to fragrec).
2. The disk cache short-circuits the subprocess: if a JSON file with the
   right hash exists, ``expand`` materialises predictions from it without
   invoking FlowER.
3. ``is_available`` is honest about a missing repo/checkpoint and
   ``expand`` raises ``FlowERUnavailable`` only when there are cache
   misses.

Run with::

    python -m pytest tests/test_flower_backend.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data.seed_species import make_EC, make_H2O, make_Li_cation
from src.flower_backend import (
    FlowERBackend,
    FlowERConfig,
    FlowERUnavailable,
    is_organic_combo,
)
from src.smiles_bridge import mols_to_dot_smiles


# ----------------------------------------------------------------------
# is_organic_combo
# ----------------------------------------------------------------------
def test_is_organic_combo_accepts_neutral_organic() -> None:
    assert is_organic_combo([make_EC(), make_H2O()]) is True


def test_is_organic_combo_rejects_lithium() -> None:
    assert is_organic_combo([make_Li_cation(), make_H2O()]) is False


def test_is_organic_combo_rejects_net_charge() -> None:
    # Two H2O + a bare Li+ is rejected for both Li and net-charge reasons;
    # we test the charge gate alone by handing in a charged but Li-free combo.
    from src.smiles_bridge import smiles_to_mol
    cation = smiles_to_mol("[NH4+]")
    assert is_organic_combo([cation]) is False


def test_is_organic_combo_rejects_radicals() -> None:
    from src.smiles_bridge import smiles_to_mol
    radical = smiles_to_mol("[CH3]")
    assert is_organic_combo([radical]) is False


# ----------------------------------------------------------------------
# Cache hit short-circuits the subprocess
# ----------------------------------------------------------------------
def _seeded_cfg(tmp_path: Path) -> FlowERConfig:
    """Configuration that points at a fresh tmp cache dir but has no FlowER
    repo/checkpoint set — so any cache miss must raise."""
    return FlowERConfig(
        repo_path=None,
        model_path=None,
        cache_dir=tmp_path / "cache",
    )


def test_cache_hit_serves_predictions_without_subprocess(tmp_path: Path) -> None:
    cfg = _seeded_cfg(tmp_path)
    backend = FlowERBackend(cfg)
    assert backend.is_available() is False  # no repo configured

    ec = make_EC()
    water = make_H2O()
    combo = (ec, water)
    smi = mols_to_dot_smiles(list(combo))

    # Pre-seed the cache with a single fake prediction:
    # EC + H2O -> H2CO3 + ethylene-glycol fragment ([OH] kept simple here).
    # We use methanol + formic acid, which is what FlowER would *plausibly*
    # produce for this combo, just to exercise the materialiser.
    fake_products = [("CO.OC=O", 25)]  # count out of sample_size=50 → P=0.5
    backend._cache_store(smi, fake_products)

    preds = backend.expand([combo])
    assert len(preds) == 1
    p = preds[0]
    assert p.probability == pytest.approx(0.5)
    assert len(p.products) == 2
    # The reactants tuple should be passed through verbatim.
    assert p.reactants == combo


def test_cache_miss_without_install_raises(tmp_path: Path) -> None:
    cfg = _seeded_cfg(tmp_path)
    backend = FlowERBackend(cfg)
    ec, water = make_EC(), make_H2O()
    with pytest.raises(FlowERUnavailable):
        backend.expand([(ec, water)])


def test_filtering_skips_non_organic_combos_silently(tmp_path: Path) -> None:
    """Even with no FlowER configured, an all-Li combo just returns []."""
    cfg = _seeded_cfg(tmp_path)
    backend = FlowERBackend(cfg)
    # Li+ + Li+ → filtered out by is_organic_combo, so no subprocess attempt.
    out = backend.expand([(make_Li_cation(), make_Li_cation())])
    assert out == []


# ----------------------------------------------------------------------
# FlowERConfig.from_env round-trip
# ----------------------------------------------------------------------
def test_from_env_picks_up_overrides(monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Path) -> None:
    repo = tmp_path / "FlowER"
    model = tmp_path / "ckpt"
    repo.mkdir()
    model.mkdir()
    monkeypatch.setenv("REPO_PATH", str(repo))
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.setenv("SAMPLE_SIZE", "25")
    monkeypatch.setenv("EMB_DIM", "128")
    cfg = FlowERConfig.from_env()
    assert cfg.repo_path == repo
    assert cfg.model_path == model
    assert cfg.sample_size == 25
    assert cfg.emb_dim == 128

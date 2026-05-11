"""Regression tests for Phase 6 (test cleanup).

Two narrow contracts worth pinning so future refactors don't quietly
break them:

1. ``get_registry()`` returns the same instance across calls, but
   ``reset_registry()`` (called by ``conftest._reset_singletons``)
   forces the next call to rebuild. If either side of this contract
   breaks, every test that depends on a fresh registry per test (e.g.
   anything that monkeypatches a source's ``is_available``) becomes
   silently order-dependent.

2. The autouse ``_snapshot_grounding_config`` fixture restores
   ``synthetic.grounding`` even when a test mutates it. Without this,
   a single misbehaving test in any file (not just
   ``test_phase4a_regressions``) would cascade into cross-file
   pollution that's miserable to diagnose.
"""

from __future__ import annotations

import copy
import os
import threading
from unittest.mock import patch

from casecrawler.config import get_config
from casecrawler.models.config import GroundingConfig
from casecrawler.sources.registry import (
    SourceRegistry,
    get_registry,
    reset_registry,
)

# Capture the *default* grounding config at import time so the part_1 /
# part_2 pair below can assert full restoration to a known baseline,
# not just inequality on one field. Deep-copied so later mutations
# inside tests can't bleed back into this reference.
_BASELINE_GROUNDING = copy.deepcopy(get_config().synthetic.grounding)
_MUTATED_GROUNDING: GroundingConfig | None = None


# ---- get_registry singleton contract ---------------------------------------


def test_get_registry_returns_same_instance_within_a_test():
    """Two calls in one test must hand back the same registry — the
    whole point of the singleton is request-handlers reusing it."""

    a = get_registry()
    b = get_registry()
    assert a is b


def test_reset_registry_forces_rebuild():
    """After ``reset_registry()``, the next ``get_registry()`` must
    return a freshly-discovered instance (NOT the cached one). This is
    what the conftest fixture relies on for per-test isolation."""

    first = get_registry()
    reset_registry()
    second = get_registry()
    assert first is not second


def test_get_registry_is_thread_safe_under_concurrent_first_access():
    """If multiple threads race the very first ``get_registry()`` call,
    they must all see the same instance — not one-per-thread caused by
    a lost race on the unlocked fast-path."""

    reset_registry()
    seen: list[SourceRegistry] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        seen.append(get_registry())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8
    assert all(s is seen[0] for s in seen)


# ---- Conftest grounding-config isolation -----------------------------------


def test_grounding_config_mutation_is_rolled_back_between_tests_part_1():
    """Together with ``part_2`` below, this pair pins the conftest's
    snapshot/restore contract. ``part_1`` flips the config to a
    different-from-baseline value; ``part_2`` asserts the next test
    sees the FULL baseline restored, not just one field unchanged."""

    global _MUTATED_GROUNDING
    cfg = get_config()
    # Pick a k that differs from the baseline so the restore is
    # observable regardless of what the default happens to be.
    mutated_k = 42 if _BASELINE_GROUNDING.k != 42 else 43
    _MUTATED_GROUNDING = GroundingConfig(
        enabled=not _BASELINE_GROUNDING.enabled,
        k=mutated_k,
    )
    cfg.synthetic.grounding = _MUTATED_GROUNDING
    assert cfg.synthetic.grounding == _MUTATED_GROUNDING


def test_grounding_config_mutation_is_rolled_back_between_tests_part_2():
    cfg = get_config()
    # Must run after part_1; otherwise the contract under test isn't
    # actually exercised.
    assert _MUTATED_GROUNDING is not None, "part_2 must run after part_1"
    # Full-field restore — catches partial-rollback regressions a
    # single-field assertion would miss.
    assert cfg.synthetic.grounding == _BASELINE_GROUNDING


# ---- In-place mutations are also rolled back -------------------------------


def test_in_place_grounding_mutation_part_1():
    """Mutating an *attribute* on the existing grounding object — rather
    than replacing the whole object — must still be rolled back. The
    conftest fixture has to deep-copy, not just save a reference."""

    cfg = get_config()
    original_enabled = cfg.synthetic.grounding.enabled
    cfg.synthetic.grounding.enabled = not original_enabled
    assert cfg.synthetic.grounding.enabled is (not original_enabled)


def test_in_place_grounding_mutation_part_2():
    """After part_1's in-place mutation, the baseline must be restored.
    If the conftest snapshot is by reference, this fails."""

    cfg = get_config()
    assert cfg.synthetic.grounding == _BASELINE_GROUNDING


# ---- get_registry honors env state at first access -------------------------


def test_get_registry_reads_env_at_first_access():
    """The singleton must be lazy enough that a test setting env vars
    BEFORE its first ``get_registry()`` call sees those vars reflected
    in the discovered source list. The conftest's ``reset_registry()``
    teardown is what makes this work — without it, the cache would
    capture env state from whichever test ran first."""

    with patch.dict(os.environ, {}, clear=True):
        reset_registry()
        registry = get_registry()
        # `pubmed` requires no key; it MUST be available.
        assert "pubmed" in registry.available_source_names

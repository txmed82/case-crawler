"""Shared pytest fixtures.

Two responsibilities:

1. Reset process-wide singletons between tests so test order can't leak
   state. Phase 5 introduced ``casecrawler.sources.registry.get_registry``
   as a lazily-built module-level cache; an earlier test that touches it
   under one env shouldn't bleed into a later test that monkeypatches
   the env. ``_reset_singletons`` runs before each test.

2. Snapshot mutable bits of the global config so tests that flip
   ``synthetic.grounding`` (today only ``test_phase4a_regressions.py``,
   but easy to add elsewhere by accident) don't leak across files.

Both fixtures are autouse, so test files don't need to opt in.
"""

from __future__ import annotations

import copy

import pytest

from casecrawler.config import get_config
from casecrawler.sources.registry import reset_registry


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Drop process-wide singletons before each test.

    Currently only the ``SourceRegistry`` cache. Add more here if new
    singletons appear — keeping the reset list in one place is cheaper
    than chasing flakiness once it starts.
    """

    reset_registry()
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def _snapshot_grounding_config():
    """Snapshot ``cfg.synthetic.grounding`` and restore on teardown.

    The synthetic pipeline reads from the global ``get_config()``
    singleton at construction time. A test that flips
    ``cfg.synthetic.grounding.enabled = True`` and forgets to undo it
    would leak into every subsequent test in the same process.
    Restoring here is cheap and removes a whole class of cross-file
    pollution.
    """

    cfg = get_config()
    # Deep copy so an in-place mutation like
    # ``cfg.synthetic.grounding.enabled = True`` is also rolled back —
    # not just attribute re-assignments.
    saved = copy.deepcopy(cfg.synthetic.grounding)
    try:
        yield
    finally:
        cfg.synthetic.grounding = saved

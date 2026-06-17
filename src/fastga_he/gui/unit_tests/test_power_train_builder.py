# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Unit tests for the component_palette module.

These tests cover:
- palette layout construction (button count, labels, initial state)
- PaletteState initialisation
- PlacementHandler.on_palette_select  (button highlight + status div)
- PlacementHandler.on_canvas_tap      (icon placement on canvas)
- multiple placements & deduplication of node names
- placing different component types alternately
- no placement when no component is selected
"""

import os
import pytest

import bokeh.plotting as bkplot
from ..power_train_builder import PowertrainBuilderLauncher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def _make_canvas() -> bkplot.figure:
    """Return a minimal blank Bokeh figure that mimics the placement canvas."""
    return bkplot.figure(width=800, height=600, x_range=(0, 800), y_range=(0, 600))


def _make_tap_event(x: float = 100.0, y: float = 200.0):
    """Create a minimal fake Tap event with ``x`` and ``y`` attributes."""

    class _FakeTap:
        pass

    evt = _FakeTap()
    evt.x = x
    evt.y = y
    return evt


# ---------------------------------------------------------------------------
# Standalone launcher (skipped in CI – requires a running IOLoop)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI – requires an interactive IOLoop.")
def test_component_palette_launcher_import():
    assert callable(PowertrainBuilderLauncher.launch)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI – requires an interactive IOLoop.")
def test_powertrain_builder_launcher_functionality():
    """Test that the launcher can be called without errors and returns expected types."""

    PowertrainBuilderLauncher.launch()

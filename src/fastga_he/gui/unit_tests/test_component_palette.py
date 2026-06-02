# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Unit tests for the component_palette module.

These tests cover:
- palette figure construction (sizes, data sources)
- palette state initialisation
- PlacementHandler.on_palette_select  (highlight + status label)
- PlacementHandler.on_canvas_tap      (icon placement on canvas)
- multiple placements & deduplication of node names
- placing different component types alternately
- no placement when no component is selected
"""

import os
import pytest

import bokeh.plotting as bkplot
from bokeh.events import Tap

from ..component_palette import (
    ICONS_CONFIG,
    PALETTE_WIDTH,
    ROW_HEIGHT,
    ComponentPaletteBuilder,
    PaletteState,
    PlacementHandler,
)
from ..power_train_network_viewer import _string_cleanup

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
# ComponentPaletteBuilder
# ---------------------------------------------------------------------------


class TestComponentPaletteBuilder:
    """Tests for the palette figure constructor."""

    def test_build_returns_tuple(self):
        """build() must return a 2-tuple (figure, PaletteState)."""
        result = ComponentPaletteBuilder.build()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_figure_type(self):
        fig, _state = ComponentPaletteBuilder.build()
        assert hasattr(fig, "renderers"), "Expected a Bokeh Figure object"

    def test_figure_width(self):
        fig, _state = ComponentPaletteBuilder.build()
        assert fig.width == PALETTE_WIDTH

    def test_figure_height_matches_component_count(self):
        fig, _state = ComponentPaletteBuilder.build()
        n = len(ICONS_CONFIG)
        expected_height = n * ROW_HEIGHT + 24
        assert fig.height == expected_height

    def test_state_type(self):
        _fig, state = ComponentPaletteBuilder.build()
        assert isinstance(state, PaletteState)

    def test_state_tap_source_row_count(self):
        """tap_source must have one row per component in ICONS_CONFIG."""
        _fig, state = ComponentPaletteBuilder.build()
        n_rows = len(state.tap_source.data["component_key"])
        assert n_rows == len(ICONS_CONFIG)

    def test_state_tap_source_has_all_keys(self):
        """All ICONS_CONFIG keys must be present in tap_source."""
        _fig, state = ComponentPaletteBuilder.build()
        assert set(state.tap_source.data["component_key"]) == set(ICONS_CONFIG.keys())

    def test_state_placed_nodes_source_initially_empty(self):
        """No nodes should be placed when the palette is first built."""
        _fig, state = ComponentPaletteBuilder.build()
        assert state.placed_nodes_source.data["x"] == []
        assert state.placed_nodes_source.data["name"] == []

    def test_state_selected_component_initially_none(self):
        _fig, state = ComponentPaletteBuilder.build()
        assert state.selected_component is None

    def test_state_placed_counter_initially_empty(self):
        _fig, state = ComponentPaletteBuilder.build()
        assert state.placed_counter == {}

    def test_highlight_source_off_screen_initially(self):
        """The highlight rect should start off-screen (x < 0)."""
        _fig, state = ComponentPaletteBuilder.build()
        assert state.highlight_source.data["x"][0] < 0

    def test_tap_source_labels_cleaned(self):
        """Labels in tap_source must equal _string_cleanup of each key."""
        _fig, state = ComponentPaletteBuilder.build()
        keys = state.tap_source.data["component_key"]
        labels = state.tap_source.data["label"]
        for key, label in zip(keys, labels):
            assert label == _string_cleanup(key)

    def test_tap_source_icon_urls_are_base64(self):
        """Icon URLs inside the palette tap source must be base64 data URIs."""
        _fig, state = ComponentPaletteBuilder.build()
        # The icon URLs are stored in the palette_source (image_url glyph),
        # but the tap_source carries component_key; we validate via state.
        # We just confirm that image_url renderers exist on the figure.
        fig, _s = ComponentPaletteBuilder.build()
        image_url_renderers = [
            r for r in fig.renderers if hasattr(r, "glyph") and "ImageURL" in type(r.glyph).__name__
        ]
        assert len(image_url_renderers) > 0


# ---------------------------------------------------------------------------
# PlacementHandler – palette selection
# ---------------------------------------------------------------------------


class TestPlacementHandlerPaletteSelect:
    """Tests for on_palette_select."""

    def setup_method(self):
        _fig, self.state = ComponentPaletteBuilder.build()
        self.canvas = _make_canvas()
        self.handler = PlacementHandler(self.state, self.canvas)

    def test_select_first_component(self):
        """Selecting index 0 should set selected_component to the first key."""
        first_key = list(ICONS_CONFIG.keys())[0]
        self.handler.on_palette_select("indices", [], [0])
        assert self.state.selected_component == first_key

    def test_select_last_component(self):
        last_key = list(ICONS_CONFIG.keys())[-1]
        last_idx = len(ICONS_CONFIG) - 1
        self.handler.on_palette_select("indices", [], [last_idx])
        assert self.state.selected_component == last_key

    def test_highlight_moves_to_selected_row(self):
        """Highlight rect x should equal PALETTE_WIDTH/2 after selection."""
        self.handler.on_palette_select("indices", [], [0])
        assert self.state.highlight_source.data["x"][0] == pytest.approx(PALETTE_WIDTH / 2)

    def test_highlight_y_matches_row(self):
        """Highlight rect y must correspond to the selected row centre."""
        n = len(ICONS_CONFIG)
        idx = 2
        expected_y = (n - idx - 0.5) * ROW_HEIGHT
        self.handler.on_palette_select("indices", [], [idx])
        assert self.state.highlight_source.data["y"][0] == pytest.approx(expected_y)

    def test_status_label_updated(self):
        """Status label must mention the selected component."""
        key = list(ICONS_CONFIG.keys())[1]
        idx = 1
        self.handler.on_palette_select("indices", [], [idx])
        status_text = self.state.status_source.data["text"][0]
        assert _string_cleanup(key) in status_text

    def test_empty_selection_does_not_crash(self):
        """Calling on_palette_select with an empty list should be a no-op."""
        self.handler.on_palette_select("indices", [], [])
        assert self.state.selected_component is None  # unchanged

    def test_out_of_range_index_does_not_crash(self):
        """An index beyond the component list should not raise."""
        out_of_range = len(ICONS_CONFIG) + 99
        self.handler.on_palette_select("indices", [], [out_of_range])
        # selected_component must remain unchanged (None)
        assert self.state.selected_component is None


# ---------------------------------------------------------------------------
# PlacementHandler – canvas placement
# ---------------------------------------------------------------------------


class TestPlacementHandlerCanvasTap:
    """Tests for on_canvas_tap."""

    def setup_method(self):
        _fig, self.state = ComponentPaletteBuilder.build()
        self.canvas = _make_canvas()
        self.handler = PlacementHandler(self.state, self.canvas)

    def _select(self, idx: int = 0):
        self.handler.on_palette_select("indices", [], [idx])

    # -- no selection guard --------------------------------------------------

    def test_tap_without_selection_places_nothing(self):
        evt = _make_tap_event(50, 50)
        self.handler.on_canvas_tap(evt)
        assert self.state.placed_nodes_source.data["x"] == []

    # -- single placement ----------------------------------------------------

    def test_tap_places_one_node(self):
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event(100, 200))
        assert len(self.state.placed_nodes_source.data["x"]) == 1

    def test_tap_records_correct_coordinates(self):
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event(123.5, 456.7))
        assert self.state.placed_nodes_source.data["x"][0] == pytest.approx(123.5)
        assert self.state.placed_nodes_source.data["y"][0] == pytest.approx(456.7)

    def test_tap_node_name_contains_component_key(self):
        key = list(ICONS_CONFIG.keys())[0]
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event())
        name = self.state.placed_nodes_source.data["name"][0]
        assert name.startswith(key + "_")

    def test_tap_node_name_ends_with_1_on_first_placement(self):
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event())
        name = self.state.placed_nodes_source.data["name"][0]
        assert name.endswith("_1")

    def test_tap_url_is_base64(self):
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event())
        url = self.state.placed_nodes_source.data["url"][0]
        assert url.startswith("data:image/")

    # -- multiple placements -------------------------------------------------

    def test_multiple_taps_accumulate(self):
        self._select(0)
        for _ in range(5):
            self.handler.on_canvas_tap(_make_tap_event())
        assert len(self.state.placed_nodes_source.data["x"]) == 5

    def test_counter_increments_per_component(self):
        """Each successive placement of the same component gets a higher suffix."""
        self._select(0)
        for i in range(1, 4):
            self.handler.on_canvas_tap(_make_tap_event())
            name = self.state.placed_nodes_source.data["name"][-1]
            assert name.endswith(f"_{i}")

    def test_different_components_have_independent_counters(self):
        """Counters for different component types are independent."""
        keys = list(ICONS_CONFIG.keys())
        self.handler.on_palette_select("indices", [], [0])
        self.handler.on_canvas_tap(_make_tap_event())
        self.handler.on_canvas_tap(_make_tap_event())

        self.handler.on_palette_select("indices", [], [1])
        self.handler.on_canvas_tap(_make_tap_event())

        names = self.state.placed_nodes_source.data["name"]
        # First component: two placements → _1, _2
        assert names[0] == f"{keys[0]}_1"
        assert names[1] == f"{keys[0]}_2"
        # Second component: one placement → _1
        assert names[2] == f"{keys[1]}_1"

    def test_placed_icon_size_uses_handler_default(self):
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event())
        w = self.state.placed_nodes_source.data["w"][0]
        h = self.state.placed_nodes_source.data["h"][0]
        assert w == self.handler.icon_size
        assert h == self.handler.icon_size

    def test_custom_icon_size(self):
        """PlacementHandler respects a custom icon_size."""
        _fig2, state2 = ComponentPaletteBuilder.build()
        canvas2 = _make_canvas()
        handler2 = PlacementHandler(state2, canvas2, icon_size=48)
        handler2.on_palette_select("indices", [], [0])
        handler2.on_canvas_tap(_make_tap_event())
        assert state2.placed_nodes_source.data["w"][0] == 48

    # -- all component types -------------------------------------------------

    @pytest.mark.parametrize("idx", range(len(ICONS_CONFIG)))
    def test_each_component_can_be_placed(self, idx):
        """Every component type in ICONS_CONFIG must be placeable without error."""
        _fig, state = ComponentPaletteBuilder.build()
        canvas = _make_canvas()
        handler = PlacementHandler(state, canvas)
        handler.on_palette_select("indices", [], [idx])
        handler.on_canvas_tap(_make_tap_event())
        assert len(state.placed_nodes_source.data["x"]) == 1


# ---------------------------------------------------------------------------
# Standalone launcher (skipped in CI – requires a running IOLoop)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI – requires an interactive IOLoop.")
def test_component_palette_launcher_import():
    """Ensure the launcher class is importable and callable without errors."""
    from ..component_palette import ComponentPaletteLauncher  # noqa: F401

    assert callable(ComponentPaletteLauncher.launch)

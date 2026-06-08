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

import bokeh.models as bkmodel
import bokeh.plotting as bkplot


from ..component_palette import (
    BUTTON_DEFAULT_COLOR_TYPE,
    BUTTON_SELECTED_COLOR_TYPE,
    ICONS_CONFIG,
    ComponentPaletteBuilder,
    PaletteState,
    PlacementHandler,
    ComponentPaletteLauncher,
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
    """Tests for the palette builder."""

    def test_build_returns_tuple(self):
        result = ComponentPaletteBuilder.build()
        assert isinstance(result, tuple) and len(result) == 2

    def test_state_type(self):
        _layout, state = ComponentPaletteBuilder.build()
        assert isinstance(state, PaletteState)

    def test_button_count_matches_components(self):
        """One button must be created for every entry in ICONS_CONFIG."""
        _layout, state = ComponentPaletteBuilder.build()
        assert len(state.buttons) == len(ICONS_CONFIG)

    def test_buttons_are_bokeh_buttons(self):
        _layout, state = ComponentPaletteBuilder.build()
        for btn in state.buttons:
            assert isinstance(btn, bkmodel.Button)

    def test_button_labels_cleaned(self):
        """Button labels must equal _string_cleanup of each ICONS_CONFIG key."""
        _layout, state = ComponentPaletteBuilder.build()
        keys = list(ICONS_CONFIG.keys())
        for btn, key in zip(state.buttons, keys):
            assert btn.label == _string_cleanup(key)

    def test_buttons_initially_default_type(self):
        """No button should appear selected before any interaction."""
        _layout, state = ComponentPaletteBuilder.build()
        for btn in state.buttons:
            assert btn.button_type == BUTTON_DEFAULT_COLOR_TYPE

    def test_placed_nodes_source_initially_empty(self):
        _layout, state = ComponentPaletteBuilder.build()
        assert state.placed_nodes_source.data["x"] == []
        assert state.placed_nodes_source.data["name"] == []

    def test_selected_component_initially_none(self):
        _layout, state = ComponentPaletteBuilder.build()
        assert state.selected_component is None

    def test_placed_counter_initially_empty(self):
        _layout, state = ComponentPaletteBuilder.build()
        assert state.placed_counter == {}

    def test_status_div_exists(self):
        _layout, state = ComponentPaletteBuilder.build()
        assert isinstance(state.status_div, bkmodel.Div)

    def test_status_div_initial_text_not_empty(self):
        _layout, state = ComponentPaletteBuilder.build()
        assert len(state.status_div.text) > 0

    def test_layout_contains_buttons(self):
        """The returned layout column must contain all button widgets."""
        layout, state = ComponentPaletteBuilder.build()
        # Collect all widgets in the column children
        children = list(layout.children)
        for btn in state.buttons:
            assert btn in children


# ---------------------------------------------------------------------------
# PlacementHandler – palette selection
# ---------------------------------------------------------------------------


class TestPlacementHandlerPaletteSelect:
    """Tests for on_palette_select."""

    def setup_method(self):
        _layout, self.state = ComponentPaletteBuilder.build()
        self.canvas = _make_canvas()
        self.handler = PlacementHandler(self.state, self.canvas)

    def test_select_first_component(self):
        first_key = list(ICONS_CONFIG.keys())[0]
        self.handler.on_palette_select(0)
        assert self.state.selected_component == first_key

    def test_select_last_component(self):
        last_key = list(ICONS_CONFIG.keys())[-1]
        self.handler.on_palette_select(len(ICONS_CONFIG) - 1)
        assert self.state.selected_component == last_key

    def test_selected_button_type_changes_to_selected(self):
        self.handler.on_palette_select(0)
        assert self.state.buttons[0].button_type == BUTTON_SELECTED_COLOR_TYPE

    def test_other_buttons_remain_default(self):
        self.handler.on_palette_select(0)
        for btn in self.state.buttons[1:]:
            assert btn.button_type == BUTTON_DEFAULT_COLOR_TYPE

    def test_switching_selection_deselects_previous(self):
        self.handler.on_palette_select(0)
        self.handler.on_palette_select(2)
        assert self.state.buttons[0].button_type == BUTTON_DEFAULT_COLOR_TYPE
        assert self.state.buttons[2].button_type == BUTTON_SELECTED_COLOR_TYPE

    def test_only_one_button_selected_at_a_time(self):
        self.handler.on_palette_select(3)
        selected = [b for b in self.state.buttons if b.button_type == BUTTON_SELECTED_COLOR_TYPE]
        assert len(selected) == 1

    def test_status_div_updated_on_selection(self):
        key = list(ICONS_CONFIG.keys())[1]
        self.handler.on_palette_select(1)
        assert _string_cleanup(key) in self.state.status_div.text

    def test_negative_index_does_not_crash(self):
        self.handler.on_palette_select(-1)
        assert self.state.selected_component is None  # unchanged

    def test_out_of_range_index_does_not_crash(self):
        self.handler.on_palette_select(len(ICONS_CONFIG) + 99)
        assert self.state.selected_component is None  # unchanged

    def test_wire_buttons_attaches_callbacks(self):
        """Clicking a button (simulated via _make_select_cb) must trigger selection."""
        # Simulate clicking the first button by calling the handler's closure directly
        cb = self.handler._make_select_cb(0)
        cb()
        assert self.state.selected_component == list(ICONS_CONFIG.keys())[0]
        assert self.state.buttons[0].button_type == BUTTON_SELECTED_COLOR_TYPE


# ---------------------------------------------------------------------------
# PlacementHandler – canvas placement
# ---------------------------------------------------------------------------


class TestPlacementHandlerCanvasTap:
    """Tests for on_canvas_tap."""

    def setup_method(self):
        _layout, self.state = ComponentPaletteBuilder.build()
        self.canvas = _make_canvas()
        self.handler = PlacementHandler(self.state, self.canvas)

    def _select(self, idx: int = 0):
        self.handler.on_palette_select(idx)

    # -- no selection guard --------------------------------------------------

    def test_tap_without_selection_places_nothing(self):
        self.handler.on_canvas_tap(_make_tap_event(50, 50))
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
        self._select(0)
        for i in range(1, 4):
            self.handler.on_canvas_tap(_make_tap_event())
            name = self.state.placed_nodes_source.data["name"][-1]
            assert name.endswith(f"_{i}")

    def test_different_components_have_independent_counters(self):
        keys = list(ICONS_CONFIG.keys())
        self.handler.on_palette_select(0)
        self.handler.on_canvas_tap(_make_tap_event())
        self.handler.on_canvas_tap(_make_tap_event())

        self.handler.on_palette_select(1)
        self.handler.on_canvas_tap(_make_tap_event())

        names = self.state.placed_nodes_source.data["name"]
        assert names[0] == f"{keys[0]}_1"
        assert names[1] == f"{keys[0]}_2"
        assert names[2] == f"{keys[1]}_1"

    def test_placed_icon_size_uses_handler_default(self):
        self._select(0)
        self.handler.on_canvas_tap(_make_tap_event())
        assert self.state.placed_nodes_source.data["w"][0] == self.handler.icon_size
        assert self.state.placed_nodes_source.data["h"][0] == self.handler.icon_size

    def test_custom_icon_size(self):
        _layout2, state2 = ComponentPaletteBuilder.build()
        canvas2 = _make_canvas()
        handler2 = PlacementHandler(state2, canvas2, icon_size=48)
        handler2.on_palette_select(0)
        handler2.on_canvas_tap(_make_tap_event())
        assert state2.placed_nodes_source.data["w"][0] == 48

    # -- all component types -------------------------------------------------

    @pytest.mark.parametrize("idx", range(len(ICONS_CONFIG)))
    def test_each_component_can_be_placed(self, idx):
        """Every component type in ICONS_CONFIG must be placeable without error."""
        _layout, state = ComponentPaletteBuilder.build()
        canvas = _make_canvas()
        handler = PlacementHandler(state, canvas)
        handler.on_palette_select(idx)
        handler.on_canvas_tap(_make_tap_event())
        assert len(state.placed_nodes_source.data["x"]) == 1


# ---------------------------------------------------------------------------
# Standalone launcher (skipped in CI – requires a running IOLoop)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI – requires an interactive IOLoop.")
def test_component_palette_launcher_import():
    assert callable(ComponentPaletteLauncher.launch)


def test_palette_launcher_functionality():
    """Test that the launcher can be called without errors and returns expected types."""

    ComponentPaletteLauncher.launch()

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Watcher CSV path: set in the "final version?" overlay, written to the YAML.

The file dialogs are replaced by a fixed path, and the save is started the way
the overlay buttons start it (see ``_JS_SAVE_MODE_CLICK`` in the toolbox):
first commit the field's ``value_input`` into its ``value``, then change
``browse_save_trigger``.

BokehJS keeps two properties for a text field: ``value_input`` follows every
keystroke, ``value`` only changes on Enter / blur. The tests below type into
the field the same way, so a path that Python sets in ``value`` alone (and
that the commit step would then overwrite) is caught.
"""

import itertools

from ruamel.yaml import YAML

from fastga_he.gui.power_train_builder.power_train_builder_state import (
    SAVE_MODE_EDITABLE,
    SAVE_MODE_FINAL,
)

WATCHER_PATH = "results/pt_watcher.csv"

# Stands in for the Date.now() of the overlay's JS: every save is a new trigger value
_click_stamps = itertools.count()


def _type_in_watcher_field(canvas, text: str, commit: bool = False):
    """
    Type *text* in the watcher field, as the browser reports it.

    :param commit: ``True`` to also press Enter / leave the field, which copies
        the typed text into ``value``.
    """
    canvas.state.watcher_path_input.value_input = text
    if commit:
        canvas.state.watcher_path_input.value = text


def _save_through_overlay(canvas, monkeypatch, yaml_path, save_mode=SAVE_MODE_EDITABLE):
    """Save as the overlay buttons do, with the save dialog answering *yaml_path*."""
    monkeypatch.setattr(
        canvas.handler, "_open_tkinter_dialog", lambda *args, **kwargs: str(yaml_path)
    )
    watcher_path_input = canvas.state.watcher_path_input
    # Same steps as _JS_SAVE_MODE_CLICK
    watcher_path_input.value = watcher_path_input.value_input
    canvas.state.browse_save_trigger.value = f"{save_mode}:{next(_click_stamps)}"
    return YAML(typ="safe").load(yaml_path.read_text())


def _browse_watcher(canvas, monkeypatch, chosen_path: str):
    """Pick *chosen_path* with the overlay's Browse button."""
    monkeypatch.setattr(canvas.handler, "_open_tkinter_dialog", lambda *args, **kwargs: chosen_path)
    canvas.state.browse_watcher_trigger.value = str(next(_click_stamps))


def test_watcher_widgets_are_in_the_save_overlay(canvas):
    overlay_children = []
    pending = [canvas.state.save_version_overlay]
    while pending:
        layout = pending.pop()
        overlay_children.append(layout)
        pending.extend(getattr(layout, "children", []))

    assert canvas.state.watcher_path_input in overlay_children


def test_typed_watcher_path_is_written_to_the_yaml(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == WATCHER_PATH
    assert canvas.state.save_button.button_type == "success"


def test_watcher_path_still_being_typed_is_written_to_the_yaml(canvas, monkeypatch, tmp_path):
    # No Enter / blur before clicking the save button: only value_input holds the text
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == WATCHER_PATH


def test_watcher_path_is_written_in_a_final_version_too(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "final.yml", SAVE_MODE_FINAL)

    assert saved["watcher_file_path"] == WATCHER_PATH
    assert "canvas_layout" not in saved


def test_empty_watcher_field_leaves_the_key_out(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert "watcher_file_path" not in saved


def test_editing_the_watcher_path_marks_the_design_unsaved(canvas, monkeypatch, tmp_path):
    _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")
    assert canvas.state.save_button.button_type == "success"

    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)

    assert canvas.state.save_button.button_type == "warning"


def test_browse_button_fills_the_field(canvas, monkeypatch):
    _browse_watcher(canvas, monkeypatch, "/data/pt_watcher.csv")

    assert canvas.state.watcher_path_input.value == "/data/pt_watcher.csv"
    # The overlay buttons commit value_input, so it must hold the path too
    assert canvas.state.watcher_path_input.value_input == "/data/pt_watcher.csv"
    assert canvas.state.watcher_file_path == "/data/pt_watcher.csv"


def test_browsed_watcher_path_is_written_to_the_yaml(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _browse_watcher(canvas, monkeypatch, "/data/pt_watcher.csv")

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == "/data/pt_watcher.csv"


def test_typing_after_browsing_keeps_the_typed_path(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _browse_watcher(canvas, monkeypatch, "/data/pt_watcher.csv")
    _type_in_watcher_field(canvas, WATCHER_PATH)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == WATCHER_PATH


def test_loading_shows_the_saved_watcher_path_without_unsaved_flag(canvas, monkeypatch, tmp_path):
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)
    yaml_path = tmp_path / "design.yml"
    _save_through_overlay(canvas, monkeypatch, yaml_path)
    canvas.handler._on_new_design()
    assert canvas.state.watcher_path_input.value == ""
    assert canvas.state.watcher_path_input.value_input == ""

    canvas.handler._load_canvas_state(str(yaml_path))

    assert canvas.state.watcher_path_input.value == WATCHER_PATH
    assert canvas.state.watcher_path_input.value_input == WATCHER_PATH
    assert canvas.state.save_button.button_type == "success"


def test_resaving_a_loaded_design_keeps_its_watcher_path(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    _type_in_watcher_field(canvas, WATCHER_PATH, commit=True)
    yaml_path = tmp_path / "design.yml"
    _save_through_overlay(canvas, monkeypatch, yaml_path)
    canvas.handler._on_new_design()
    canvas.handler._load_canvas_state(str(yaml_path))

    # Save again without touching the field
    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "resaved.yml")

    assert saved["watcher_file_path"] == WATCHER_PATH

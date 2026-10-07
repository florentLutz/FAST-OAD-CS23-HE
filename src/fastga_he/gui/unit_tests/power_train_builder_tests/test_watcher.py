# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Watcher CSV path: set in the "final version?" overlay, written to the YAML.

The file dialogs are replaced by a fixed path, and the save is started the way
the overlay buttons start it: by changing ``browse_save_trigger``.
"""

from ruamel.yaml import YAML

from ..power_train_builder_state import SAVE_MODE_EDITABLE, SAVE_MODE_FINAL


def _save_through_overlay(canvas, monkeypatch, yaml_path, save_mode=SAVE_MODE_EDITABLE):
    """Save as the overlay buttons do, with the save dialog answering *yaml_path*."""
    monkeypatch.setattr(
        canvas.handler, "_open_tkinter_dialog", lambda *args, **kwargs: str(yaml_path)
    )
    canvas.state.browse_save_trigger.value = f"{save_mode}:{len(str(yaml_path))}"
    return YAML(typ="safe").load(yaml_path.read_text())


def test_watcher_widgets_are_in_the_save_overlay(canvas):
    overlay_children = []
    pending = [canvas.state.save_version_overlay]
    while pending:
        layout = pending.pop()
        overlay_children.append(layout)
        pending.extend(getattr(layout, "children", []))

    assert canvas.state.watcher_path_input in overlay_children


def test_watcher_path_from_the_field_is_written_to_the_yaml(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    canvas.state.watcher_path_input.value = "results/pt_watcher.csv"

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert saved["watcher_file_path"] == "results/pt_watcher.csv"
    assert canvas.state.save_button.button_type == "success"


def test_watcher_path_is_written_in_a_final_version_too(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)
    canvas.state.watcher_path_input.value = "results/pt_watcher.csv"

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "final.yml", SAVE_MODE_FINAL)

    assert saved["watcher_file_path"] == "results/pt_watcher.csv"
    assert "canvas_layout" not in saved


def test_empty_watcher_field_leaves_the_key_out(canvas, monkeypatch, tmp_path):
    canvas.place("battery", 100, 100)

    saved = _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")

    assert "watcher_file_path" not in saved


def test_editing_the_watcher_path_marks_the_design_unsaved(canvas, monkeypatch, tmp_path):
    _save_through_overlay(canvas, monkeypatch, tmp_path / "design.yml")
    assert canvas.state.save_button.button_type == "success"

    canvas.state.watcher_path_input.value = "results/pt_watcher.csv"

    assert canvas.state.save_button.button_type == "warning"


def test_browse_button_fills_the_field(canvas, monkeypatch):
    monkeypatch.setattr(
        canvas.handler, "_open_tkinter_dialog", lambda *args, **kwargs: "/data/pt_watcher.csv"
    )

    canvas.state.browse_watcher_trigger.value = "1"

    assert canvas.state.watcher_path_input.value == "/data/pt_watcher.csv"
    assert canvas.state.watcher_file_path == "/data/pt_watcher.csv"


def test_loading_shows_the_saved_watcher_path_without_unsaved_flag(canvas, monkeypatch, tmp_path):
    canvas.state.watcher_path_input.value = "results/pt_watcher.csv"
    yaml_path = tmp_path / "design.yml"
    _save_through_overlay(canvas, monkeypatch, yaml_path)
    canvas.handler._on_new_design()
    assert canvas.state.watcher_path_input.value == ""

    canvas.handler._load_canvas_state(str(yaml_path))

    assert canvas.state.watcher_path_input.value == "results/pt_watcher.csv"
    assert canvas.state.save_button.button_type == "success"

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""Unit tests for the config panel's module-level helpers."""

import pytest

from fastga_he.gui.power_train_builder.power_train_builder_handler_config_panel import (
    PRIMARY_PORT_LABEL_COLOR,
    PRIMARY_SOURCE_PORT_TYPE_KEYWORDS,
    PRIMARY_TARGET_PORT_TYPE_KEYWORDS,
    _option_values_to_strings,
    _port_label_stylesheets,
    _strings_to_option_values,
)


@pytest.mark.parametrize("value", [True, False, 3, 2.5, "percent_split"])
def test_option_value_round_trips_through_display_string(value):
    """
    Test that the all options values can be converted to a display string and back to the same
    value.
    """
    display_string = _option_values_to_strings(value)

    assert isinstance(display_string, str)
    assert _strings_to_option_values(display_string) == value
    assert type(_strings_to_option_values(display_string)) is type(value)


@pytest.mark.parametrize(
    "node_type, port_label, keywords",
    [
        ("dc_splitter", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS),
        ("gearbox", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS),
        ("planetary_gear", "1", PRIMARY_TARGET_PORT_TYPE_KEYWORDS),
    ],
)
def test_primary_port_label_is_highlighted(node_type, port_label, keywords):
    stylesheets = _port_label_stylesheets(node_type, port_label, keywords)

    assert len(stylesheets) == 1
    assert PRIMARY_PORT_LABEL_COLOR in stylesheets[0]


@pytest.mark.parametrize(
    "node_type, port_label, keywords",
    [
        ("dc_splitter", "2", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS),
        ("dc_splitter", "1", PRIMARY_TARGET_PORT_TYPE_KEYWORDS),
        ("planetary_gear", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS),
        ("pmsm", "1", PRIMARY_SOURCE_PORT_TYPE_KEYWORDS),
    ],
)
def test_other_port_labels_keep_default_style(node_type, port_label, keywords):
    assert _port_label_stylesheets(node_type, port_label, keywords) == []

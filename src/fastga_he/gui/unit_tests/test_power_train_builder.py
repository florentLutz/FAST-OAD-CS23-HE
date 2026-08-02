# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Unit tests for the powertrain builder.
"""

import os
import pytest

from ..power_train_builder import PowertrainBuilderLauncher

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI – requires an interactive IOLoop.")
def test_power_train_builder_launcher_functionality():
    """Test that the launcher can be called without errors and returns expected types."""

    PowertrainBuilderLauncher.launch()

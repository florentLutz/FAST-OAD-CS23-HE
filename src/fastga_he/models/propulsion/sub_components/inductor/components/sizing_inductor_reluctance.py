# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorReluctance(om.ExplicitComponent):
    """
    Computation of the reluctance of the filter inductor, implementation of the formula from
    :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]
        settings_prefix = prefix.replace("data", "settings")

        self.add_input(
            name=prefix + ":inductor:air_gap",
            units="m",
            val=np.nan,
            desc="Air gap in the inductor",
        )
        self.add_input(
            name=settings_prefix + ":inductor:iron_permeability",
            units="uH/m",
            val=4.396,
            desc="Permeability of the iron core",
        )
        self.add_input(
            name=prefix + ":inductor:iron_surface",
            units="m**2",
            val=np.nan,
            desc="Iron surface of the E-core inductor",
        )

        self.add_output(
            name=prefix + ":inductor:reluctance",
            units="H**-1",
            val=2.00e9,
            desc="Reluctance of the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]
        settings_prefix = prefix.replace("data", "settings")

        mu = inputs[settings_prefix + ":inductor:iron_permeability"] * 1e-6
        air_gap = inputs[prefix + ":inductor:air_gap"]
        iron_area = inputs[prefix + ":inductor:iron_surface"]

        reluctance = 2.0 * air_gap / (mu * iron_area)

        outputs[prefix + ":inductor:reluctance"] = reluctance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]
        settings_prefix = prefix.replace("data", "settings")

        mu = inputs[settings_prefix + ":inductor:iron_permeability"] * 1e-6
        air_gap = inputs[prefix + ":inductor:air_gap"]
        iron_area = inputs[prefix + ":inductor:iron_surface"]

        partials[
            prefix + ":inductor:reluctance",
            settings_prefix + ":inductor:iron_permeability",
        ] = -2.0 * air_gap / (mu**2.0 * iron_area) * 1e-6
        partials[
            prefix + ":inductor:reluctance",
            prefix + ":inductor:air_gap",
        ] = 2.0 / (mu * iron_area)
        partials[prefix + ":inductor:reluctance", prefix + ":inductor:iron_surface"] = (
            -2.0 * air_gap / (mu * iron_area**2.0)
        )

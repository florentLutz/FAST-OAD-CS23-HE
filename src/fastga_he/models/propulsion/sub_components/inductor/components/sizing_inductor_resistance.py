# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorResistance(om.ExplicitComponent):
    """
    Computation of the resistance the inductor, based on the electric resistivity of copper
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
            prefix + ":inductor:core_dimension:B",
            units="m",
            val=np.nan,
            desc="B dimension of the E-core in the inductor",
        )
        self.add_input(
            prefix + ":inductor:core_dimension:C",
            units="m",
            val=np.nan,
            desc="C dimension of the E-core in the inductor",
        )
        self.add_input(
            name=prefix + ":inductor:turn_number",
            val=np.nan,
            desc="Number of turns in the inductor",
        )
        self.add_input(
            name=settings_prefix + ":inductor:copper_resistivity",
            units="ohm*um",
            val=1.77e-2,
            desc="Resistivity of the copper used in the inductor [Ohm/m]",
        )
        self.add_input(
            name=prefix + ":inductor:wire_section_area",
            units="cm**2",
            val=np.nan,
            desc="Section area of the copper wire inside the inductor",
        )

        self.add_output(
            name=prefix + ":inductor:resistance",
            units="ohm",
            val=4e-3,
            desc="Resistance of the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]
        settings_prefix = prefix.replace("data", "settings")

        b_length = inputs[prefix + ":inductor:core_dimension:B"]
        c_length = inputs[prefix + ":inductor:core_dimension:C"]
        turn_number = inputs[prefix + ":inductor:turn_number"]
        copper_resistivity = inputs[settings_prefix + ":inductor:copper_resistivity"] / 1e6
        wire_section_area = inputs[prefix + ":inductor:wire_section_area"] / 1e4

        resistance = (
            2.0
            * np.pi
            * (b_length + c_length)
            / 4.0
            * turn_number
            * copper_resistivity
            / wire_section_area
        )

        outputs[prefix + ":inductor:resistance"] = resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]
        settings_prefix = prefix.replace("data", "settings")

        b_length = inputs[prefix + ":inductor:core_dimension:B"]
        c_length = inputs[prefix + ":inductor:core_dimension:C"]
        turn_number = inputs[prefix + ":inductor:turn_number"]
        copper_resistivity = inputs[settings_prefix + ":inductor:copper_resistivity"] / 1e6
        wire_section_area = inputs[prefix + ":inductor:wire_section_area"] / 1e4

        partials[
            prefix + ":inductor:resistance",
            prefix + ":inductor:core_dimension:B",
        ] = 2.0 * np.pi / 4.0 * turn_number * copper_resistivity / wire_section_area
        partials[
            prefix + ":inductor:resistance",
            prefix + ":inductor:core_dimension:C",
        ] = 2.0 * np.pi / 4.0 * turn_number * copper_resistivity / wire_section_area
        partials[
            prefix + ":inductor:resistance",
            prefix + ":inductor:turn_number",
        ] = 2.0 * np.pi * (b_length + c_length) / 4.0 * copper_resistivity / wire_section_area
        partials[
            prefix + ":inductor:resistance",
            settings_prefix + ":inductor:copper_resistivity",
        ] = 2.0 * np.pi * (b_length + c_length) / 4.0 * turn_number / wire_section_area / 1e6
        partials[
            prefix + ":inductor:resistance",
            prefix + ":inductor:wire_section_area",
        ] = -(
            2.0
            * np.pi
            * (b_length + c_length)
            / 4.0
            * turn_number
            * copper_resistivity
            / wire_section_area**2.0
            / 1e4
        )

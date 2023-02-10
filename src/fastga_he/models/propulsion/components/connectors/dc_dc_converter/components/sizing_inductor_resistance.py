# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterInductorResistance(om.ExplicitComponent):
    """
    Computation of the resistance the inductor, based on the electric resistivity of copper
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:B",
            units="m",
            val=np.nan,
            desc="B dimension of the E-core in the inductor",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:C",
            units="m",
            val=np.nan,
            desc="C dimension of the E-core in the inductor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:turn_number",
            val=np.nan,
            desc="Number of turns in the inductor",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_resistivity",
            units="ohm/m",
            val=1.77e-8,
            desc="Resistivity of the copper used in the inductor [Ohm/m]",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
            units="m**2",
            val=np.nan,
            desc="Section area of the copper wire inside the inductor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            units="ohm",
            val=4e-3,
            desc="Resistance of the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        b_length = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:B"
        ]
        c_length = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:C"
        ]
        turn_number = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:turn_number"
        ]
        copper_resistivity = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_resistivity"
        ]
        wire_section_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area"
        ]

        resistance = (
            2.0
            * np.pi
            * (b_length + c_length)
            / 4.0
            * turn_number
            * copper_resistivity
            / wire_section_area
        )

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance"
        ] = resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        b_length = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:B"
        ]
        c_length = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:C"
        ]
        turn_number = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:turn_number"
        ]
        copper_resistivity = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_resistivity"
        ]
        wire_section_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:B",
        ] = (
            2.0 * np.pi / 4.0 * turn_number * copper_resistivity / wire_section_area
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:C",
        ] = (
            2.0 * np.pi / 4.0 * turn_number * copper_resistivity / wire_section_area
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:turn_number",
        ] = (
            2.0 * np.pi * (b_length + c_length) / 4.0 * copper_resistivity / wire_section_area
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_resistivity",
        ] = (
            2.0 * np.pi * (b_length + c_length) / 4.0 * turn_number / wire_section_area
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
        ] = -(
            2.0
            * np.pi
            * (b_length + c_length)
            / 4.0
            * turn_number
            * copper_resistivity
            / wire_section_area ** 2.0
        )

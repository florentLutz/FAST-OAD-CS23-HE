# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterInductorCopperMass(om.ExplicitComponent):
    """
    Computation of the copper mass for the inductor, implementation of the formula from
    :cite:`budinger_sizing_2023`.
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
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
            units="m**2",
            val=np.nan,
            desc="Section area of the copper wire inside the inductor",
        )
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
            + ":inductor:copper_density",
            units="kg/m**3",
            val=7800.0,
            desc="Density of the copper used in the inductor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass",
            units="kg",
            val=1.0,
            desc="Mass of the copper in the inductor",
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
        wire_section_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area"
        ]
        copper_density = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_density"
        ]

        copper_weight = (
            2.0
            * np.pi
            * (b_length + c_length)
            / 4.0
            * turn_number
            * wire_section_area
            * copper_density
        )

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass"
        ] = copper_weight

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
        wire_section_area = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area"
        ]
        copper_density = inputs[
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_density"
        ]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:B",
        ] = (
            2.0 * np.pi / 4.0 * turn_number * wire_section_area * copper_density
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:core_dimension:C",
        ] = (
            2.0 * np.pi / 4.0 * turn_number * wire_section_area * copper_density
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:turn_number",
        ] = (
            2.0 * np.pi * (b_length + c_length) / 4.0 * wire_section_area * copper_density
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
        ] = (
            2.0 * np.pi * (b_length + c_length) / 4.0 * turn_number * copper_density
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_mass",
            "settings:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:copper_density",
        ] = (
            2.0 * np.pi * (b_length + c_length) / 4.0 * turn_number * wire_section_area
        )

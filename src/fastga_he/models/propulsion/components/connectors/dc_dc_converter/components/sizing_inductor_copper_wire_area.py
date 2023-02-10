# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterInductorCopperWireArea(om.ExplicitComponent):
    """
    Computation of the copper wire section area for the inductor, implementation of the formula from
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
            + ":inductor:current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the DC/DC converter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_current_density",
            units="A/m**2",
            val=5e6,
            desc="Maximum current density allowed in the copper wire",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
            units="m**2",
            val=100e-6,
            desc="Section area of the copper wire inside the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area"
        ] = (
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:current_caliber"
            ]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:wire_current_density"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_caliber",
        ] = (
            1.0
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:wire_current_density"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_section_area",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:wire_current_density",
        ] = -(
            inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:current_caliber"
            ]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:wire_current_density"
            ]
            ** 2.0
        )

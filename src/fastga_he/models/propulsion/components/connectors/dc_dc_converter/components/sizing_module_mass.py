# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCDCConverterCasingWeight(om.ExplicitComponent):
    """
    Computation of the weight of the diode and IGBT, which includes 3 IGBT modules plus their
    casing. Based on a regression on the SEMIKRON family from :cite:`giraud:2014`.
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
            + ":module:current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the converter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:mass",
            units="kg",
            val=0.350,
            desc="Weight of the diode and IGBT module",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        module_current_caliber = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":module:mass"
        ] = (0.175 + 4e-4 * module_current_caliber)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":module:mass",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":module:current_caliber",
        ] = 4e-4

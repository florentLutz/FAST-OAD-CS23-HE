# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBusBarCrossSectionDimensions(om.ExplicitComponent):
    """
    Computation of the bus bar cross section dimensions, based on a width to thickness ratio.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area",
            units="cm**2",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:w_t_ratio",
            val=20,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
            units="cm",
            val=1,
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
            units="cm",
            val=1,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        cross_section_area = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area"
        ]
        w_t_ratio = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:w_t_ratio"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness"
        ] = np.sqrt(cross_section_area / w_t_ratio)
        outputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width"
        ] = np.sqrt(cross_section_area * w_t_ratio)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        cross_section_area = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area"
        ]
        w_t_ratio = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:w_t_ratio"
        ]

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area",
        ] = (
            1.0 / 2.0 * np.sqrt(1.0 / (w_t_ratio * cross_section_area))
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:w_t_ratio",
        ] = (
            -1.0 / 2.0 * np.sqrt(cross_section_area / w_t_ratio ** 3.0)
        )

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area",
        ] = (
            1.0 / 2.0 * np.sqrt(w_t_ratio / cross_section_area)
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:w_t_ratio",
        ] = (
            1.0 / 2.0 * np.sqrt(cross_section_area / w_t_ratio)
        )

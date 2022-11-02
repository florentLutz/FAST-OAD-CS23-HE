# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBusBarCrossSectionArea(om.ExplicitComponent):
    """
    Computation of the cross section of the conductor plate of the bus bar. We will assume a
    single conductor plate and its grounding plate for now and will adapt the formula for this
    case (as in not adding the corresponding input).

    Based on the formula from :cite:`khan:2014`.
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
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
            units="A",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area",
            units="cm**2",
            val=1,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        max_current = inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max"]
        number_of_plates = 1.0

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area"] = (
            400.0 * max_current * 0.785 * (1 + 0.05 * (number_of_plates - 1.0)) * 1e-6 / 0.155
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        number_of_plates = 1.0

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:area",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
        ] = (
            400.0 * 0.785 * (1 + 0.05 * (number_of_plates - 1.0)) * 1e-6 / 0.155
        )

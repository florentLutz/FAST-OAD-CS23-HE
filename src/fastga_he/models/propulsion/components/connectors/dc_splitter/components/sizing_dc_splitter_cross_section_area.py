# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCSplitterCrossSectionArea(om.ExplicitComponent):
    """
    Computation of the cross section of the conductor plate of the splitter. We will take the
    same assumption as for the bus bar.

    Based on the formula from :cite:`khan:2014`.
    """

    def initialize(self):

        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":current_caliber",
            units="A",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":cross_section:area",
            units="cm**2",
            val=1,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]

        max_current = inputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_caliber"
        ]
        number_of_plates = 1.0

        outputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:area"
        ] = (400.0 * max_current * 0.785 * (1 + 0.05 * (number_of_plates - 1.0)) * 1e-6 / 0.155)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]

        number_of_plates = 1.0

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":cross_section:area",
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_caliber",
        ] = (
            400.0 * 0.785 * (1 + 0.05 * (number_of_plates - 1.0)) * 1e-6 / 0.155
        )

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

COPPER_RESISTIVITY_20 = 1.68e-8  # Copper resistivity at 20°C [Ohm·m]
COPPER_TEMPERATURE_COEFF = 0.00393  # Temperature coefficient for copper [1/°C]


class SizingWindingResistivity(om.ExplicitComponent):
    """
    Computation of the Winding resistivity.

    """

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_temperature",
            val=np.nan,
            units="degC",
        )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity",
            units="ohm*m",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=COPPER_RESISTIVITY_20 * COPPER_TEMPERATURE_COEFF)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":resistivity"] = (
            COPPER_RESISTIVITY_20
            * (
                1.0
                + COPPER_TEMPERATURE_COEFF
                * (
                    inputs[
                        "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_temperature"
                    ]
                    - 20.0
                )
            )
        )

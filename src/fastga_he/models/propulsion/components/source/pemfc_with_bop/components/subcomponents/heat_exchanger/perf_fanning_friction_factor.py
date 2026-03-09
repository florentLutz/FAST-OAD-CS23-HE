# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesFanningFrictionFactor(om.ExplicitComponent):
    """
    Computation of the fanning friction factor for both fluid in the heat exchanger. Surrogate model
    based on the Reynolds number of the flow, with a transition at 1500, obtained from
    Valentine's thesis.
    """

    def setup(self):
        self.add_input(
            name="air_reynolds_number",
            units="unitless",
            val=np.nan,
        )
        self.add_input(
            name="coolant_reynolds_number",
            units="unitless",
            val=np.nan,
        )

        self.add_output(
            name="air_fanning_friction_factor",
            units="unitless",
            val=0.42,
        )
        self.add_output(
            name="coolant_fanning_friction_factor",
            units="unitless",
            val=0.088,
        )

    def setup_partials(self):
        self.declare_partials("air_fanning_friction_factor", "air_reynolds_number", method="exact")
        self.declare_partials(
            "coolant_fanning_friction_factor", "coolant_reynolds_number", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]

        outputs["air_fanning_friction_factor"] = np.where(
            air_reynolds_number < 1500,
            6.04 * air_reynolds_number**-0.68,
            0.36 * air_reynolds_number**-0.28,
        )
        outputs["coolant_fanning_friction_factor"] = np.where(
            air_reynolds_number < 1500,
            6.04 * coolant_reynolds_number**-0.68,
            0.36 * coolant_reynolds_number**-0.28,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        air_reynolds_number = inputs["air_reynolds_number"]
        coolant_reynolds_number = inputs["coolant_reynolds_number"]

        partials["air_fanning_friction_factor", "air_reynolds_number"] = np.where(
            air_reynolds_number < 1500,
            -4.1072 * air_reynolds_number**-1.68,
            -0.1008 * air_reynolds_number**-1.28,
        )

        partials["coolant_fanning_friction_factor", "coolant_reynolds_number"] = np.where(
            coolant_reynolds_number < 1500,
            -4.1072 * coolant_reynolds_number**-1.68,
            -0.1008 * coolant_reynolds_number**-1.28,
        )

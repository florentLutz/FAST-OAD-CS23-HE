# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class perfMaxSpecificPower(om.ExplicitComponent):
    """Computation of the Maximum specific power."""

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("specific_power", units="W/kg", val=np.nan, shape=number_of_points)

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":max_spec_power",
            units="W/kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":max_spec_power",
            wrt="*",
            method="fd",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":max_spec_power"] = np.max(
            inputs["specific_power"]
        )

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":shaft_power_max",
    #         "shaft_power_out",
    #     ] = np.where(inputs["specific_power"] == np.max(inputs["specific_power"]), 1.0, 0.0)

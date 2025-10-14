# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesJouleLosses(om.ExplicitComponent):
    """
    Computation of the motor Joule losses due to ohmic heating in conductors. This is obtained
    from part II.3.1 in :cite:`touhami:2020`
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_in_one_phase",
            units="A",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance",
            val=np.full(number_of_points, np.nan),
            units="ohm",
        )

        self.add_output(
            "joule_power_losses",
            units="W",
            val=0.0,
            shape=number_of_points,
        )

        self.declare_partials(
            of="joule_power_losses",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        i_rms = inputs["ac_current_rms_in_one_phase"]
        resistance = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance"]

        outputs["joule_power_losses"] = 3.0 * resistance * i_rms**2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        i_rms = inputs["ac_current_rms_in_one_phase"]
        resistance = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance"]

        partials[
            "joule_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistance",
        ] = 3.0 * i_rms**2.0

        partials[
            "joule_power_losses",
            "ac_current_rms_in_one_phase",
        ] = 6.0 * resistance * i_rms

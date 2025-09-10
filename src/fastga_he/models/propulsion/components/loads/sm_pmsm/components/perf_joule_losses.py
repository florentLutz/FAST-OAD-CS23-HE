# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesJouleLosses(om.ExplicitComponent):
    """
    Computation of the motor joule losses due to ohmic heating in conductors. This is obtained
    from part II.3.1 in :cite:`touhami:2020`
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_in_one_phase",
            units="A",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":number_of_phases",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses",
            wrt=["ac_current_rms_in_one_phase"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":number_of_phases",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        i_rms = inputs["ac_current_rms_in_one_phase"]
        r_s = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance"]
        q = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":number_of_phases"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses"] = (
            q * r_s * i_rms**2.0 / 1000.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        i_rms = inputs["ac_current_rms_in_one_phase"]
        r_s = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance"]
        q = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":number_of_phases"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":resistance",
        ] = q * i_rms**2.0 / 1000.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses",
            "ac_current_rms_in_one_phase",
        ] = q * 2.0 * r_s * i_rms / 1000.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":joule_power_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":number_of_phases",
        ] = r_s * i_rms**2.0 / 1000.0

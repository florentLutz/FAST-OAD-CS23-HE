# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorSection(om.ExplicitComponent):
    """
    Computation of the Conducor section.

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
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_section",
            val=np.nan,
            units="m**2",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductor_section",
            units="m**2"
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductor_section",
            wrt=[
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_section",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        S_slot = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_section"]
        k_fill = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor"]
        k_sc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor"]

        S_cond = S_slot * k_sc * k_fill


        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductor_section"] = S_cond

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     S_slot = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_section"]
    #     k_fill = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor"]
    #     k_sc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor"]
    #
    #
    #
    #     S_cond = S_slot * k_sc * k_fill
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductor_section",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_section",
    #     ] = k_fill*k_sc
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductor_section",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
    #     ] = S_slot*k_sc
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductor_section",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor",
    #     ] = S_slot*k_fill


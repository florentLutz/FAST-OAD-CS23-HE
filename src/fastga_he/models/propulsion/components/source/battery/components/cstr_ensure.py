# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_BATTERY_SOC

import openmdao.api as om
import numpy as np

import fastoad.api as oad


oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_BATTERY_SOC
] = "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.ensure"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_BATTERY_SOC,
    "fastga_he.submodel.propulsion.constraints.battery.state_of_charge.ensure",
)
class ConstraintsSOCEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the minimum SOC seen by the battery during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):

        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
            val=np.nan,
            units="percent",
            desc="Minimum state-of-charge of the battery during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
            val=np.nan,
            units="percent",
            desc="Minimum state-of-charge that the battery can have without degradation",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":min_safe_SOC",
            val=0.0,
            units="percent",
            desc="Constraints on minimum battery SOC, respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":min_safe_SOC",
            wrt=[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "constraints:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":min_safe_SOC"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC"
            ]
            - inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "constraints:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":min_safe_SOC",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":min_safe_SOC",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":min_safe_SOC",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
        ] = -1.0

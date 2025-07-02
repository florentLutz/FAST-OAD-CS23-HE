#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCABatteryDepthOfDischarge(om.ExplicitComponent):
    """
    Computation of the depth of discharge of the battery. In that context it refers to the
    amplitude of the battery discharge on one cycle and not 100. - SOC
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
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_mission_start",
            val=100.0,
            units="percent",
            desc="State-of-Charge of the battery at the start of the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":SOC_end_main_route",
            units="percent",
            val=np.nan,
            desc="State of charge at the end of the main route (excludes reserve)",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route",
            units="percent",
            val=80.0,
            desc="Depth of battery discharge on one cycle",
        )

    def setup_partials(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":SOC_mission_start",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":SOC_end_main_route",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":DOD_main_route"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":SOC_mission_start"
            ]
            - inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":SOC_end_main_route"
            ]
        )

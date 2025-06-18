#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO


import openmdao.api as om
import numpy as np


class PerformancesSOCEndMainRoute(om.ExplicitComponent):
    """
    Identification of the state of charge at the end of the main route.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Default is set as None, because at group level if the option isn't set this component
        # shouldn't be added, so it is a second safety
        self.options.declare(
            "number_of_points_reserve",
            default=None,
            desc="number of equilibrium to be treated in reserve",
            types=int,
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":SOC_end_main_route",
            units="percent",
            val=20,
            desc="State of charge at the end of the main route (excludes reserve)",
        )

        partials_soc = np.zeros(number_of_points)
        partials_soc[-number_of_points_reserve - 2] = 1

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":SOC_end_main_route",
            wrt="state_of_charge",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=partials_soc,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points_reserve = self.options["number_of_points_reserve"]
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_end_main_route"
        ] = inputs["state_of_charge"][-number_of_points_reserve - 2]

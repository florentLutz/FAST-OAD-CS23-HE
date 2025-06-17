# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesFuelConsumedMainRoute(om.ExplicitComponent):
    """
    Computation of the amount of fuel in that particular tank which will be consumed during the
    main route, this quantity is of importance because for the LCA the fuel for reserve should not
    be included.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Default is set as None, because at group level if the option isn't set this component
        # shouldn't be added so it is a second safety
        self.options.declare(
            "number_of_points_reserve",
            default=None,
            desc="number of equilibrium to be treated in reserve",
            types=int,
        )
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            "fuel_consumed_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            desc="Fuel from this tank consumed at each time step",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:"
            + fuel_tank_id
            + ":fuel_consumed_main_route",
            units="kg",
            val=50.0,
            desc="Amount of fuel from that tank which will be consumed during the main route (does "
            "not account for takeoff and initial climb, the amount used for sizing does)",
        )

        val_partial = np.ones(number_of_points)
        val_partial[-number_of_points_reserve - 1 : -1] = np.zeros(number_of_points_reserve)

        self.declare_partials(
            of="*",
            wrt="*",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=val_partial,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        # TODO: this way of "extracting" the reserve will only work with the current "format" for
        #  the points in the mission (1pt taxi_out -> climb -> cruise -> descent -> 1pt taxi_in)
        outputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_main_route"
        ] = np.sum(inputs["fuel_consumed_t"]) - np.sum(
            inputs["fuel_consumed_t"][-number_of_points_reserve - 1 : -1]
        )

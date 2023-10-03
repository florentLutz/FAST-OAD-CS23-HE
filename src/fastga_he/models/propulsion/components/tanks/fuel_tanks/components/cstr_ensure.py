# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_FUEL_TANK_CAPACITY


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_FUEL_TANK_CAPACITY,
    "fastga_he.submodel.propulsion.constraints.fuel_tank.capacity.ensure",
)
class ConstraintsFuelTankCapacityEnsure(om.ExplicitComponent):
    """
    Class that ensures that the capacity of the tank is greater than the amount of fuel needed for
    the mission (which includes the unusable fuel).
    """

    def initialize(self):

        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):

        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_total_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of fuel loaded in the tank for the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
            units="kg",
            val=-0.0,
            desc="Constraints on the tank capacity in kg, respected if <0",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        outputs["constraints:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity"] = (
            inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_total_mission"
            ]
            - inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        partials[
            "constraints:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_total_mission",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":capacity",
        ] = -1.0

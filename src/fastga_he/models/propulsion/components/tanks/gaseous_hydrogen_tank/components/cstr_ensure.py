# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_GASEOUS_HYDROGEN_TANK_CAPACITY


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_GASEOUS_HYDROGEN_TANK_CAPACITY,
    "fastga_he.submodel.propulsion.constraints.gaseous_hydrogen_tank.capacity.ensure",
)
class ConstraintsGaseousHydrogenTankCapacityEnsure(om.ExplicitComponent):
    """
    Class that ensures that the capacity of the tank is greater than the amount of fuel needed for
    the mission (which includes the unusable fuel).
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of hydrogen gas loaded in the tank for the mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            units="kg",
            val=-0.0,
            desc="Constraints on the tank capacity in kg, respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            val=1.0,
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        outputs[
            "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity"
        ] = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_total_mission"
            ]
            - inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":capacity"
            ]
        )

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_CRYOGENIC_HYDROGEN_TANK_CAPACITY

# This choice was made. "Why ? Because I can" (Katarina from LoL)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_CRYOGENIC_HYDROGEN_TANK_CAPACITY] = (
    "fastga_he.submodel.propulsion.constraints.cryogenic_hydrogen_tank.capacity.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_CRYOGENIC_HYDROGEN_TANK_CAPACITY,
    "fastga_he.submodel.propulsion.constraints.cryogenic_hydrogen_tank.capacity.enforce",
)
class ConstraintsCryogenicHydrogenTankCapacityEnforce(om.ExplicitComponent):
    """
    Class that enforces that the capacity of the tank is equal to the amount of fuel needed for
    the mission (which includes the unusable hydrogen and boil-off hydrogen).
    """

    def initialize(self):
        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":fuel_total_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of hydrogen loaded in the tank for the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":capacity",
            units="kg",
            val=15.15,
            desc="Capacity of the tank in terms of weight",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]

        outputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":capacity"
        ] = inputs[
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":fuel_total_mission"
        ]

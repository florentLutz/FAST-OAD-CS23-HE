#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

import fastoad.api as oad

from fastga_he.exceptions import ControlParameterInconsistentShapeError

from ..constants import SUBMODEL_PMSM_EFFICIENCY


@oad.RegisterSubmodel(
    SUBMODEL_PMSM_EFFICIENCY, "fastga_he.submodel.propulsion.pmsm.efficiency.fixed"
)
class PerformancesFixedEfficiencyMission(om.ExplicitComponent):
    """
    Component which takes the desired efficiency for the PMSM operation from the data and gives
    it the right format for the mission. Assume the value doesn't depend on operating
    conditions, another submodel exists where efficiency is computed based on the known formula
    for pmsm losses.

    The input efficiency can either be a float (then during the whole mission the frequency is
    going to be the same) or an array of number of points elements for the individual control of
    each point.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":efficiency_mission",
            val=np.nan,
            desc="Efficiency of the PMSM for the points",
            shape_by_conn=True,
        )

        self.add_output("efficiency", val=0.98, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        eta_mission = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":efficiency_mission"
        ]

        if len(eta_mission) == 1:
            outputs["efficiency"] = np.full(number_of_points, eta_mission)

        elif len(eta_mission) == number_of_points:
            outputs["efficiency"] = eta_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:PMSM:"
                + motor_id
                + ":efficiency_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        eta_mission = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":efficiency_mission"
        ]

        if len(eta_mission) == 1:
            partials[
                "efficiency",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":efficiency_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "efficiency",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":efficiency_mission",
            ] = sp.sparse.eye(number_of_points, format="csc")

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

import fastoad.api as oad

from fastga_he.exceptions import ControlParameterInconsistentShapeError

from ..constants import SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE

SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE_FIXED = (
    "fastga_he.submodel.propulsion.rectifier.junction_temperature.fixed"
)
oad.RegisterSubmodel.active_models[SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE] = (
    SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE_FIXED
)


@oad.RegisterSubmodel(
    SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE, SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE_FIXED
)
class PerformancesJunctionTemperatureMission(om.ExplicitComponent):
    """
    Component which takes the desired junction temperature for rectifier operation from the data
    and gives it the right format for the mission. Assume the value doesn't depend from operating
    conditions, another submodel exists where temperatures are computed based on the losses and
    heat sink temperature.

    The input junction temperature can either be a float (then during the whole mission the
    temperature is going to be the same) or an array of number of points elements for
    the individual control of each point.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":junction_temperature_mission",
            val=np.nan,
            units="degK",
            desc="Junction temperature of the rectifier for the performances computation",
            shape_by_conn=True,
        )

        self.add_output(
            "diode_temperature",
            val=np.full(number_of_points, 273.15),
            units="degK",
            desc="Temperature of the diodes inside the module",
            shape=number_of_points,
        )
        self.add_output(
            "IGBT_temperature",
            val=np.full(number_of_points, 273.15),
            units="degK",
            desc="Temperature of the IGBTs inside the module",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        t_j_mission = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":junction_temperature_mission"
        ]

        if len(t_j_mission) == 1:
            outputs["diode_temperature"] = np.full(number_of_points, t_j_mission)
            outputs["IGBT_temperature"] = np.full(number_of_points, t_j_mission)

        elif len(t_j_mission) == number_of_points:
            outputs["diode_temperature"] = t_j_mission
            outputs["IGBT_temperature"] = t_j_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":junction_temperature_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        t_j_mission = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":junction_temperature_mission"
        ]

        if len(t_j_mission) == 1:
            partials[
                "diode_temperature",
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":junction_temperature_mission",
            ] = np.full(number_of_points, 1.0)
            partials[
                "IGBT_temperature",
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":junction_temperature_mission",
            ] = np.full(number_of_points, 1.0)

        elif len(t_j_mission) == number_of_points:
            partials[
                "diode_temperature",
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":junction_temperature_mission",
            ] = sp.sparse.eye(number_of_points, format="csc")
            partials[
                "IGBT_temperature",
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":junction_temperature_mission",
            ] = sp.sparse.eye(number_of_points, format="csc")

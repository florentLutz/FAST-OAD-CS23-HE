# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.command.api import list_inputs_metadata
from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)


class OperationalMissionVectorWithTargetSoC(OperationalMissionVector):
    def initialize(self):

        super().initialize()

        self.options.declare(
            "variable_name_target_SoC",
            types=str,
            default=None,
            allow_none=False,
            desc="Name of the variable that will be used to evaluate if target SOC is reached",
        )

    def setup(self):

        super().setup()
        self.add_subsystem(
            name="distance_to_target",
            subsys=DistanceToTargetSoc(
                variable_name_target_SoC=self.options["variable_name_target_SoC"]
            ),
            promotes=["*"],
        )


class DistanceToTargetSoc(om.ImplicitComponent):
    def initialize(self):

        self.options.declare(
            "variable_name_target_SoC",
            types=str,
            default=None,
            allow_none=False,
            desc="Name of the variable that will be used to evaluate if target SOC is reached",
        )

    def setup(self):

        variable_name_target_soc = self.options["variable_name_target_SoC"]

        self.add_input(variable_name_target_soc, val=np.nan, units="percent")
        self.add_input("data:mission:payload_range:threshold_SoC", val=np.nan, units="percent")

        self.add_output("data:mission:operational:range", units="NM", val=30.0)

        self.declare_partials(
            of="data:mission:operational:range", wrt=variable_name_target_soc, val=1.0
        )
        self.declare_partials(
            of="data:mission:operational:range",
            wrt="data:mission:payload_range:threshold_SoC",
            val=-1.0,
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        variable_name_target_soc = self.options["variable_name_target_SoC"]

        residuals["data:mission:operational:range"] = (
            inputs[variable_name_target_soc] - inputs["data:mission:payload_range:threshold_SoC"]
        )


def zip_op_mission_input_from_soc(pt_file_path):
    """
    Returns a list of the variables needed for the computation of the equilibrium. Based on
    the submodel currently registered and the propulsion_id required.

    :param pt_file_path: Path to the powertrain file.
    :return inputs_zip: a zip containing a list of name, a list of units, a list of shapes,
    a list of shape_by_conn boolean and a list of copy_shape str.
    """

    new_component = AutoUnitsDefaultGroup()
    new_component.add_subsystem(
        "system",
        OperationalMissionVector(
            number_of_points_climb=30,
            number_of_points_cruise=30,
            number_of_points_descent=20,
            number_of_points_reserve=10,
            power_train_file_path=pt_file_path,
            pre_condition_pt=True,
            use_linesearch=False,
        ),
        promotes=["*"],
    )

    name, unit, value, shape, shape_by_conn, copy_shape = list_inputs_metadata(new_component)
    input_zip = zip(name, unit, value, shape, shape_by_conn, copy_shape)

    return input_zip

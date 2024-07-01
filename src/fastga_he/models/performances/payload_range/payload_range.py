# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import logging
import time

import openmdao.api as om
import numpy as np

import fastoad.api as oad
from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.command.api import list_inputs_metadata
from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)
from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)


class ComputePayloadRange(om.ExplicitComponent):
    """
    Computation of the characteristic points of the payload-range diagram. Will use the
    operational mission module and inputs to compute the different points.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

        self._input_zip = None
        self.cached_problem = None

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )

    def setup(self):

        # I'm not really happy with doing it here, but for that model to work we need to ensure
        # those submodels are active
        oad.RegisterSubmodel.active_models[
            HE_SUBMODEL_ENERGY_CONSUMPTION
        ] = "fastga_he.submodel.performances.energy_consumption.from_pt_file"
        oad.RegisterSubmodel.active_models[
            HE_SUBMODEL_DEP_EFFECT
        ] = "fastga_he.submodel.performances.dep_effect.from_pt_file"

        self.configurator.load(self.options["power_train_file_path"])

        self.zip_op_mission_input(self.options["power_train_file_path"])

        for (
            var_names,
            var_unit,
            var_value,
            var_shape,
            var_shape_by_conn,
            var_copy_shape,
        ) in self._input_zip:
            var_prefix = var_names.split(":")[0]
            if var_prefix == "data" or var_prefix == "settings" or var_prefix == "convergence":
                if var_shape_by_conn:
                    self.add_input(
                        name=var_names,
                        val=np.nan,
                        units=var_unit,
                        shape_by_conn=var_shape_by_conn,
                        copy_shape=var_copy_shape,
                    )
                else:
                    self.add_input(
                        name=var_names,
                        val=var_value,
                        units=var_unit,
                        shape=var_shape,
                    )

        self.add_input("data:mission:payload_range:threshold_SoC", val=np.nan, units="percent")

        self.add_output("data:mission:payload_range:range", val=1.0, units="NM")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        self.zip_op_mission_input(self.options["power_train_file_path"])

        ivc = om.IndepVarComp()
        for var_names, var_unit, _, _, _, _ in self._input_zip:
            var_prefix = var_names.split(":")[0]
            if var_prefix == "data" or var_prefix == "settings" or var_prefix == "convergence":
                if var_names != "data:mission:operational:range":
                    ivc.add_output(
                        name=var_names,
                        val=inputs[var_names],
                        units=var_unit,
                        shape=np.shape(inputs[var_names]),
                    )

        # Add it manually
        ivc.add_output(
            name="data:mission:payload_range:threshold_SoC",
            val=inputs["data:mission:payload_range:threshold_SoC"],
            units="percent",
            shape=np.shape(inputs["data:mission:payload_range:threshold_SoC"]),
        )

        self.cached_problem = om.Problem()
        model = self.cached_problem.model

        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        model.add_subsystem(
            "op_mission",
            OperationalMissionVectorWithTargetSoC(
                number_of_points_climb=30,
                number_of_points_cruise=30,
                number_of_points_descent=20,
                number_of_points_reserve=10,
                power_train_file_path=self.options["power_train_file_path"],
                pre_condition_pt=True,
                use_linesearch=False,
                use_apply_nonlinear=False,
                variable_name_target_SoC="data:propulsion:he_power_train:battery_pack:battery_pack_2:SOC_min",
            ),
            promotes=["*"],
        )

        # Replace the old solver with a NewtonSolver to handle the ImplicitComponent
        # TODO: It would be better to straight up a NewtonSolver in the OperationalMissionVector
        #  class but that would required to check if it is at all possible first
        model.op_mission.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.op_mission.nonlinear_solver.options["iprint"] = 0
        model.op_mission.nonlinear_solver.options["maxiter"] = 100
        model.op_mission.nonlinear_solver.options["rtol"] = 1e-5
        model.op_mission.nonlinear_solver.options["atol"] = 1e-5
        model.op_mission.nonlinear_solver.options["stall_limit"] = 2
        model.op_mission.nonlinear_solver.options["stall_tol"] = 1e-5
        model.op_mission.linear_solver = om.DirectSolver()

        self.cached_problem.setup()
        self.cached_problem.run_model()

        print(self.cached_problem.get_val("data:mission:operational:range", units="NM"))
        print(
            self.cached_problem.get_val(
                "data:propulsion:he_power_train:battery_pack:battery_pack_2:SOC_min",
                units="percent",
            )
        )

    def zip_op_mission_input(self, pt_file_path):
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
        self._input_zip = zip(name, unit, value, shape, shape_by_conn, copy_shape)


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

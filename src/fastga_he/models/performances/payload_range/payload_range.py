# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO


import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)

from .mission_range_from_soc import (
    OperationalMissionVectorWithTargetSoC,
    zip_op_mission_input_from_soc,
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

        self._input_zip = zip_op_mission_input_from_soc(self.options["power_train_file_path"])

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

        self._input_zip = zip_op_mission_input_from_soc(self.options["power_train_file_path"])

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

        target_range = self.cached_problem.get_val("data:mission:operational:range", units="NM")

        outputs["data:mission:payload_range:range"] = target_range

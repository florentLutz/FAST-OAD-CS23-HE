# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO


import openmdao.api as om
import numpy as np

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain
from fastoad.openmdao.problem import AutoUnitsDefaultGroup

from fastga_he.command.api import list_inputs_metadata
from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator, PT_DATA_PREFIX

from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)
from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)

from .mission_range_from_soc import OperationalMissionVectorWithTargetSoC
from .mission_range_from_fuel import OperationalMissionVectorWithTargetFuel


@oad.RegisterOpenMDAOSystem("fastga_he.payload_range.outer", domain=ModelDomain.PERFORMANCE)
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

        self._input_zip = zip_op_mission_input(self.options["power_train_file_path"])

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

        if self.configurator.will_aircraft_mass_vary():
            tank_names, tank_types = self.configurator.get_fuel_tank_list()
            for tank_name, tank_type in zip(tank_names, tank_types):
                self.add_input(
                    name=PT_DATA_PREFIX + tank_type + ":" + tank_name + ":capacity",
                    val=np.nan,
                    units="kg",
                )

        else:
            self.add_input("data:mission:payload_range:threshold_SoC", val=np.nan, units="percent")

        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")

        self.add_input(
            "data:mission:payload_range:carbon_intensity_fuel",
            val=3.81,
            desc="Carbon intensity of the fuel in kgCO2 per kg of fuel",
        )
        self.add_input(
            "data:mission:payload_range:carbon_intensity_electricity", val=72.7, units="g/MJ"
        )

        self.add_output("data:mission:payload_range:range", val=1.0, units="NM", shape=4)
        self.add_output("data:mission:payload_range:payload", val=1.0, units="kg", shape=4)
        self.add_output(
            "data:mission:payload_range:emission_factor",
            val=1.0,
            shape=4,
            desc="Emission factor in kgCO2 per kg of payload per km",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        self._input_zip = zip_op_mission_input(self.options["power_train_file_path"])

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
        if not self.configurator.will_aircraft_mass_vary():
            ivc.add_output(
                name="data:mission:payload_range:threshold_SoC",
                val=inputs["data:mission:payload_range:threshold_SoC"],
                units="percent",
                shape=np.shape(inputs["data:mission:payload_range:threshold_SoC"]),
            )
        else:
            tank_names, tank_types = self.configurator.get_fuel_tank_list()
            mfw = 0.0
            for tank_name, tank_type in zip(tank_names, tank_types):
                mfw += inputs[PT_DATA_PREFIX + tank_type + ":" + tank_name + ":capacity"]

        self.cached_problem = om.Problem()
        model = self.cached_problem.model

        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        if self.configurator.will_aircraft_mass_vary():
            model.add_subsystem(
                "op_mission",
                OperationalMissionVectorWithTargetFuel(
                    number_of_points_climb=30,
                    number_of_points_cruise=30,
                    number_of_points_descent=20,
                    number_of_points_reserve=10,
                    power_train_file_path=self.options["power_train_file_path"],
                    pre_condition_pt=True,
                    use_linesearch=False,
                    use_apply_nonlinear=False,
                ),
                promotes=["*"],
            )
        else:
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
                    variable_name_target_SoC="data:propulsion:he_power_train:battery_pack:battery_pack_1:SOC_min",
                ),
                promotes=["*"],
            )

        # Replace the old solver with a NewtonSolver to handle the ImplicitComponent
        model.op_mission.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.op_mission.nonlinear_solver.options["iprint"] = 0
        model.op_mission.nonlinear_solver.options["maxiter"] = 100
        model.op_mission.nonlinear_solver.options["rtol"] = 1e-5
        model.op_mission.nonlinear_solver.options["atol"] = 1e-5
        model.op_mission.nonlinear_solver.options["stall_limit"] = 2
        model.op_mission.nonlinear_solver.options["stall_tol"] = 1e-5
        model.op_mission.linear_solver = om.DirectSolver()

        self.cached_problem.setup()

        # There are four points in the payload range as computed by this framework. Points A, B,
        # D and E. Yes, I know.
        range_array = np.zeros(4)
        payload_array = np.zeros(4)
        ef_array = np.zeros(4)

        max_payload = inputs["data:weight:aircraft:max_payload"][0]
        mtow = inputs["data:weight:aircraft:MTOW"][0]
        owe = inputs["data:weight:aircraft:OWE"][0]

        carbon_intensity_fuel = inputs["data:mission:payload_range:carbon_intensity_fuel"]
        carbon_intensity_electricity = (
            inputs["data:mission:payload_range:carbon_intensity_electricity"] / 1000.0
        )  # In kgCO2 per MJ

        # Point A correspond to max payload, no range. The emission factor is not really defined
        # here since emissions are nil but so is the distances
        payload_array[0] = max_payload
        ef_array[0] = 0

        # Point B correspond to max payload and the range that leads to the MTOW. On an electric
        # aircraft this correspond to the design point
        payload_array[1] = max_payload
        self.cached_problem.set_val(
            "data:mission:operational:payload:mass",
            max_payload,
            units="kg",
        )
        if self.configurator.will_aircraft_mass_vary():
            target_fuel = max(mtow - owe - max_payload, 0.0)
            self.cached_problem.set_val(
                "data:mission:payload_range:target_fuel", target_fuel, units="kg"
            )
            self.cached_problem.run_model()
            range_point_b = self.cached_problem.get_val(
                "data:mission:operational:range", units="NM"
            )[0]
            range_array[1] = range_point_b
        else:
            # The threshold value is already set we can just go ahead and compute the corresponding
            # range
            self.cached_problem.run_model()
            range_point_b = self.cached_problem.get_val(
                "data:mission:operational:range", units="NM"
            )[0]
            range_array[1] = range_point_b

        emissions_point_b = (
            self.cached_problem.get_val("data:mission:operational:fuel", units="kg")[0]
            * carbon_intensity_fuel
            + self.cached_problem.get_val("data:mission:operational:energy", units="kW*h")[0]
            * 3.6
            * carbon_intensity_electricity
        )
        emission_factor_b = emissions_point_b / (range_point_b * 1.852) / max_payload
        ef_array[1] = emission_factor_b

        # Point D corresponds to MFW and MTOW. On an electric aircraft this point is the same as
        # the previous one, so we will simply not recompute it.
        if self.configurator.will_aircraft_mass_vary():
            payload = mtow - mfw - owe
            self.cached_problem.set_val(
                "data:mission:operational:payload:mass",
                payload,
                units="kg",
            )
            self.cached_problem.set_val("data:mission:payload_range:target_fuel", mfw, units="kg")
            self.cached_problem.run_model()
            range_point_d = self.cached_problem.get_val(
                "data:mission:operational:range", units="NM"
            )[0]
            payload_array[2] = payload
            range_array[2] = range_point_d

            emissions_point_d = (
                self.cached_problem.get_val("data:mission:operational:fuel", units="kg")[0]
                * carbon_intensity_fuel
                + self.cached_problem.get_val("data:mission:operational:energy", units="kW*h")[0]
                * 3.6
                * carbon_intensity_electricity
            )
            emission_factor_d = emissions_point_d / (range_point_d * 1.852) / payload
            ef_array[2] = emission_factor_d
        else:
            payload_array[2] = payload_array[1]
            range_array[2] = range_array[1]
            ef_array[2] = ef_array[1]

        # Point E correspond to no payload and MFW. On an electric aircraft, it's simply no
        # payload. Here, the emission factor is not defined as the payload goes to 0 so we will
        # have an emission factor of 0.
        self.cached_problem.set_val(
            "data:mission:operational:payload:mass",
            0.0,
            units="kg",
        )
        self.cached_problem.run_model()
        range_point_e = self.cached_problem.get_val("data:mission:operational:range", units="NM")[0]
        payload_array[3] = 0.0
        range_array[3] = range_point_e
        ef_array[3] = 0.0

        outputs["data:mission:payload_range:range"] = range_array
        outputs["data:mission:payload_range:payload"] = payload_array
        outputs["data:mission:payload_range:emission_factor"] = ef_array


def zip_op_mission_input(pt_file_path):
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

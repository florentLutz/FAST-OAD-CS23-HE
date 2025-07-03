# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
from fastga_he.models.performances.op_mission_vector.op_mission_vector import (
    OperationalMissionVector,
)
from .payload_range import zip_op_mission_input
from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)

INVALID_COMPUTATION_RESULT = -1.0


class ComputePayloadRangeInner(om.ExplicitComponent):
    """
    Computation of the performances of the aircraft on some points inside the payload range diagram.
    Will use the operational mission module and inputs to compute the different points.
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
        oad.RegisterSubmodel.active_models[HE_SUBMODEL_ENERGY_CONSUMPTION] = (
            "fastga_he.submodel.performances.energy_consumption.from_pt_file"
        )
        oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT] = (
            "fastga_he.submodel.performances.dep_effect.from_pt_file"
        )

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

        self.add_input(
            "data:mission:payload_range:carbon_intensity_fuel",
            val=3.81,
            desc="Carbon intensity of the fuel in kgCO2 per kg of fuel",
        )
        self.add_input(
            "data:mission:payload_range:carbon_intensity_electricity", val=72.7, units="g/MJ"
        )

        self.add_input("data:mission:payload_range:range", val=np.nan, units="NM", shape=4)
        self.add_input("data:mission:payload_range:payload", val=np.nan, units="kg", shape=4)

        self.add_input(
            "data:mission:inner_payload_range:range", val=np.nan, units="NM", shape_by_conn=True
        )
        self.add_input(
            "data:mission:inner_payload_range:payload", val=np.nan, units="kg", shape_by_conn=True
        )

        self.add_output(
            "data:mission:inner_payload_range:fuel",
            val=1.0,
            units="kg",
            shape_by_conn=True,
            copy_shape="data:mission:inner_payload_range:range",
            desc="Fuel consumed during selected mission",
        )
        self.add_output(
            "data:mission:inner_payload_range:energy",
            val=1.0,
            units="kW*h",
            shape_by_conn=True,
            copy_shape="data:mission:inner_payload_range:range",
            desc="Fuel consumed during selected mission",
        )
        self.add_output(
            "data:mission:inner_payload_range:emissions",
            val=1.0,
            units="kg",
            shape_by_conn=True,
            copy_shape="data:mission:inner_payload_range:range",
            desc="Emissions of the aircraft in kgCO2",
        )
        self.add_output(
            "data:mission:inner_payload_range:emission_factor",
            val=1.0,
            shape_by_conn=True,
            copy_shape="data:mission:inner_payload_range:range",
            desc="Emission factor in kgCO2 per kg of payload per km",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inner_payload_array = inputs["data:mission:inner_payload_range:payload"]
        inner_range_array = inputs["data:mission:inner_payload_range:range"]

        outer_payload_array = inputs["data:mission:payload_range:payload"]
        outer_range_array = inputs["data:mission:payload_range:range"]

        carbon_intensity_fuel = inputs["data:mission:payload_range:carbon_intensity_fuel"]
        carbon_intensity_electricity = (
            inputs["data:mission:payload_range:carbon_intensity_electricity"] / 1000.0
        )  # In kgCO2 per MJ

        inner_fuel_array = np.zeros_like(inner_payload_array)
        inner_energy_array = np.zeros_like(inner_payload_array)
        inner_emissions_array = np.zeros_like(inner_payload_array)
        inner_emission_factor_array = np.zeros_like(inner_payload_array)

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

        self.cached_problem = om.Problem(reports=False)
        model = self.cached_problem.model

        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
        model.add_subsystem(
            "op_mission",
            OperationalMissionVector(
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

        self.cached_problem.setup()

        for idx, range_value in enumerate(inner_range_array):
            if self.is_in_payload_range_envelope(
                payload_envelope=outer_payload_array,
                range_envelope=outer_range_array,
                payload_point=inner_payload_array[idx],
                range_point=range_value,
            ):
                self.cached_problem.set_val(
                    "data:mission:operational:payload:mass",
                    inner_payload_array[idx],
                    units="kg",
                )
                self.cached_problem.set_val(
                    "data:mission:operational:range",
                    range_value,
                    units="NM",
                )
                self.cached_problem.run_model()

                fuel_that_mission = self.cached_problem.get_val(
                    "data:mission:operational:fuel", units="kg"
                )[0]
                energy_that_mission = self.cached_problem.get_val(
                    "data:mission:operational:energy", units="kW*h"
                )[0]
                emission_that_mission = (
                    fuel_that_mission * carbon_intensity_fuel
                    + energy_that_mission * 3.6 * carbon_intensity_electricity
                )
                emission_factor_that_mission = (
                    emission_that_mission / inner_payload_array[idx] / (range_value * 1.852)
                )

                inner_fuel_array[idx] = fuel_that_mission
                inner_energy_array[idx] = energy_that_mission
                inner_emissions_array[idx] = emission_that_mission.item()
                inner_emission_factor_array[idx] = emission_factor_that_mission.item()

            else:
                inner_fuel_array[idx] = INVALID_COMPUTATION_RESULT
                inner_energy_array[idx] = INVALID_COMPUTATION_RESULT
                inner_emissions_array[idx] = INVALID_COMPUTATION_RESULT
                inner_emission_factor_array[idx] = INVALID_COMPUTATION_RESULT

        outputs["data:mission:inner_payload_range:fuel"] = inner_fuel_array
        outputs["data:mission:inner_payload_range:energy"] = inner_energy_array
        outputs["data:mission:inner_payload_range:emissions"] = inner_emissions_array
        outputs["data:mission:inner_payload_range:emission_factor"] = inner_emission_factor_array

    @staticmethod
    def is_in_payload_range_envelope(
        payload_envelope: np.ndarray,
        range_envelope: np.ndarray,
        payload_point: float,
        range_point: float,
    ) -> bool:
        """
        Check that the input payload range point is inside the envelope.
        """

        if payload_point > max(payload_envelope):
            return False
        else:
            max_range_that_payload = np.interp(
                payload_point, np.flip(payload_envelope[1:]), np.flip(range_envelope[1:])
            )
            if range_point > max_range_that_payload:
                return False
            else:
                return True

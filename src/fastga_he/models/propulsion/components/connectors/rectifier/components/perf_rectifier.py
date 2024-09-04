# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .perf_voltage_out_target import PerformancesVoltageOutTargetMission
from .perf_switching_frequency import PerformancesSwitchingFrequencyMission
from .perf_heat_sink_temperature import PerformancesHeatSinkTemperatureMission
from .perf_modulation_index import PerformancesModulationIndex
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_resistance import PerformancesResistance
from .perf_gate_voltage import PerformancesGateVoltage
from .perf_conduction_loss import PerformancesConductionLosses
from .perf_total_loss import PerformancesLosses
from .perf_casing_temperature import PerformancesCasingTemperature
from .perf_load_side import PerformancesRectifierLoadSide
from .perf_generator_side import PerformancesRectifierGeneratorSide
from .perf_rectifier_relations import PerformancesRectifierRelations
from .perf_maximum import PerformancesMaximum
from .perf_junction_temperature_fixed import SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE_FIXED

from ..constants import SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE, SUBMODEL_RECTIFIER_EFFICIENCY


class PerformancesRectifier(om.Group):
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

        self.add_subsystem(
            "voltage_out_target",
            PerformancesVoltageOutTargetMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "switching_frequency",
            PerformancesSwitchingFrequencyMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "heat_sink_temperature",
            PerformancesHeatSinkTemperatureMission(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "modulation_idx",
            PerformancesModulationIndex(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "load_side",
            PerformancesRectifierLoadSide(number_of_points=number_of_points),
            promotes=["ac_voltage_rms_in", "ac_current_rms_in_one_phase"],
        )
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "temperature_profile",
            PerformancesRectifierTemperature(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "generator_side",
            PerformancesRectifierGeneratorSide(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "dc_current_out"],
        )

        efficiency_option = {
            "rectifier_id": rectifier_id,
            "number_of_points": number_of_points,
        }
        self.add_subsystem(
            "efficiency",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_RECTIFIER_EFFICIENCY, options=efficiency_option
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "converter_relation",
            PerformancesRectifierRelations(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "voltage_out_target", "dc_current_out", "efficiency"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(number_of_points=number_of_points, rectifier_id=rectifier_id),
            promotes=["*"],
        )

        self.connect("converter_relation.power_rel", "load_side.power")
        self.connect("converter_relation.voltage_out_rel", "generator_side.voltage_target")

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        # One easy thing we can do is pre-read the value we will set as a target for the voltage
        # and set it there

        not_formatted_voltage = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":voltage_out_target_mission"
        ]
        formatted_voltage = format_to_array(not_formatted_voltage, number_of_points)

        outputs["voltage_out_target.voltage_out_target"] = formatted_voltage


class PerformancesRectifierTemperature(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup, if temperature is fixed, there is no loop so we can omit it
        if (
            oad.RegisterSubmodel.active_models[SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE]
            != SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE_FIXED
        ):
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            self.nonlinear_solver.options["iprint"] = 0
            self.nonlinear_solver.options["maxiter"] = 50
            self.nonlinear_solver.options["rtol"] = 1e-5
            self.nonlinear_solver.options["stall_limit"] = 10
            self.nonlinear_solver.options["stall_tol"] = 1e-5
            self.linear_solver = om.DirectSolver()

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

        junction_temperature_option = {
            "rectifier_id": rectifier_id,
            "number_of_points": number_of_points,
        }

        self.add_subsystem(
            "junction_temperature",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE, options=junction_temperature_option
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "resistance_profile",
            PerformancesResistance(number_of_points=number_of_points, rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "gate_voltage_profile",
            PerformancesGateVoltage(number_of_points=number_of_points, rectifier_id=rectifier_id),
            promotes=["*"],
        )
        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "casing_temperature",
            PerformancesCasingTemperature(
                number_of_points=number_of_points, rectifier_id=rectifier_id
            ),
            promotes=["*"],
        )


def format_to_array(input_array: np.ndarray, number_of_points: int) -> np.ndarray:
    """
    Takes an inputs which is either a one-element array or a multi-element array and formats it.
    """

    if len(input_array):
        output_array = np.full(number_of_points, input_array[0])
    else:
        output_array = input_array

    return output_array

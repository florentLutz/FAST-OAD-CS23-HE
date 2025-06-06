# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .perf_switching_frequency import PerformancesSwitchingFrequencyMission
from .perf_voltage_out_target import PerformancesVoltageOutTargetMission
from .perf_converter_relations import PerformancesConverterRelations
from .perf_generator_side import PerformancesConverterGeneratorSide
from .perf_load_side import PerformancesConverterLoadSide
from .perf_duty_cycle import PerformancesDutyCycle
from .perf_currents import PerformancesCurrents
from .perf_switching_losses import PerformancesSwitchingLosses
from .perf_conduction_losses import PerformancesConductionLosses
from .perf_total_losses import PerformancesLosses
from .perf_dc_power_in import PerformancesDCPowerIn
from .perf_maximum import PerformancesMaximum

from ..constants import SUBMODEL_DC_DC_CONVERTER_EFFICIENCY


class PerformancesDCDCConverter(om.Group):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_subsystem(
            "switching_frequency",
            PerformancesSwitchingFrequencyMission(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "voltage_out_target",
            PerformancesVoltageOutTargetMission(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "load_side",
            PerformancesConverterLoadSide(number_of_points=number_of_points),
            promotes=["dc_voltage_in", "dc_current_in"],
        )
        self.add_subsystem(
            "generator_side",
            PerformancesConverterGeneratorSide(number_of_points=number_of_points),
            promotes=["dc_voltage_out", "dc_current_out"],
        )
        self.add_subsystem(
            "duty_cycle",
            PerformancesDutyCycle(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "currents",
            PerformancesCurrents(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "switching_losses",
            PerformancesSwitchingLosses(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "conduction_losses",
            PerformancesConductionLosses(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "total_losses",
            PerformancesLosses(number_of_points=number_of_points),
            promotes=["*"],
        )
        option_efficiency = {
            "number_of_points": number_of_points,
            "dc_dc_converter_id": dc_dc_converter_id,
        }
        self.add_subsystem(
            "efficiency",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_DC_DC_CONVERTER_EFFICIENCY, options=option_efficiency
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "dc_power_in",
            PerformancesDCPowerIn(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum",
            PerformancesMaximum(
                number_of_points=number_of_points, dc_dc_converter_id=dc_dc_converter_id
            ),
            promotes=[
                "*",
            ],
        )
        self.add_subsystem(
            "converter_relation",
            PerformancesConverterRelations(
                number_of_points=number_of_points,
            ),
            promotes=["dc_voltage_out", "voltage_out_target", "efficiency"],
        )

        self.connect("converter_relation.power_rel", "load_side.power")
        self.connect("dc_current_out", "converter_relation.dc_current_out")
        self.connect("converter_relation.voltage_out_rel", "generator_side.voltage_target")

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        # One easy thing we can do is pre-read the value we will set as a target for the voltage
        # and set it there

        not_formatted_voltage = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_target_mission"
        ]
        formatted_voltage = format_to_array(not_formatted_voltage, number_of_points)

        outputs["voltage_out_target.voltage_out_target"] = formatted_voltage


def format_to_array(input_array: np.ndarray, number_of_points: int) -> np.ndarray:
    """
    Takes an inputs which is either a one-element array or a multi-element array and formats it.
    """

    if len(input_array):
        output_array = np.full(number_of_points, input_array[0])
    else:
        output_array = input_array

    return output_array

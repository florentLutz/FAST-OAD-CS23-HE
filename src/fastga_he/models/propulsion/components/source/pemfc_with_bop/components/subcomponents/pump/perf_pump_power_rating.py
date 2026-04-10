# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPumpPower(om.ExplicitComponent):
    """
    The required power of the pump.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pump_id",
            default=None,
            desc="Identifier of the pump",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pump_id = self.options["pump_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pressure_compensation",
            units="Pa",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":motor_efficiency",
            units="unitless",
            val=0.9,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pump_efficiency",
            units="unitless",
            val=0.65,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate",
            units="m**3/s",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            units="W",
            val=10.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pump_id = self.options["pump_id"]

        pressure_compensation = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pressure_compensation"
        ]
        motor_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":motor_efficiency"
        ]
        pump_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pump_efficiency"
        ]
        volumetric_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate"
        ]

        unclipped_power_required = (
            volumetric_flow_rate * pressure_compensation / (pump_efficiency * motor_efficiency)
        )

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating"
        ] = np.clip(unclipped_power_required, 0.0, 50000.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pump_id = self.options["pump_id"]

        pressure_compensation = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pressure_compensation"
        ]
        motor_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":motor_efficiency"
        ]
        pump_efficiency = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pump_efficiency"
        ]
        volumetric_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate"
        ]

        unclipped_power_required = (
            volumetric_flow_rate * pressure_compensation / (pump_efficiency * motor_efficiency)
        )
        clipped_required_power = np.clip(unclipped_power_required, 0.0, 50000.0)
        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pressure_compensation",
        ] = np.where(
            unclipped_power_required == clipped_required_power,
            volumetric_flow_rate / (pump_efficiency * motor_efficiency),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":motor_efficiency",
        ] = np.where(
            unclipped_power_required == clipped_required_power,
            -volumetric_flow_rate
            * pressure_compensation
            / (pump_efficiency * motor_efficiency**2.0),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":pump_efficiency",
        ] = np.where(
            unclipped_power_required == clipped_required_power,
            -volumetric_flow_rate
            * pressure_compensation
            / (pump_efficiency**2.0 * motor_efficiency),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate",
        ] = np.where(
            unclipped_power_required == clipped_required_power,
            pressure_compensation / (pump_efficiency * motor_efficiency),
            1e-6,
        )

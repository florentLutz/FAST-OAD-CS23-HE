# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om


class PerformancesPEMFCStackBOPCoolantTemperature(om.ExplicitComponent):
    """
    Coolant temperature computation of the PEMFC stack, using for BOP and TMS sizing.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "coolant_temperature_gradiant",
            default=10.0,
            desc="The temperature difference of the PEMFC coolant I/O [K]",
        )
        self.options.declare(
            "pemfc_heat_transfer_effectiveness",
            default=0.98,
            desc="The heat transfer effectiveness of PEMFC",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":operating_temperature",
            units="K",
            val=350,
            desc="standard operating temperature for the PEMFC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:inlet_temperature",
            units="K",
            val=300.0,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature",
            units="K",
            val=310.0,
        )
        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mean_temperature",
            units="K",
            val=305.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        temperature_gradiant = self.options["coolant_temperature_gradiant"]
        heat_transfer_effectiveness = self.options["pemfc_heat_transfer_effectiveness"]
        operating_temperature = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":operating_temperature"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:inlet_temperature"
        ] = operating_temperature - (temperature_gradiant / heat_transfer_effectiveness)

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:outlet_temperature"
        ] = (
            operating_temperature
            - (temperature_gradiant / heat_transfer_effectiveness)
            + temperature_gradiant
        )

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mean_temperature"
        ] = (
            operating_temperature
            - (temperature_gradiant / heat_transfer_effectiveness)
            + (temperature_gradiant / 2.0)
        )

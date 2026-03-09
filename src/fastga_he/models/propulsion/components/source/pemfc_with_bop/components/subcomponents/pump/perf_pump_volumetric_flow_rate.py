# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPumpVolumetricFlowRate(om.ExplicitComponent):
    """
    The design volumetric flow rate of the pump.
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
            name="coolant_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
            units="kg/s",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":safety_factor",
            units="unitless",
            val=1.2,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate",
            units="m**3/s",
            val=1e-4,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pump_id = self.options["pump_id"]

        density = inputs["coolant_density"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        safety_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":safety_factor"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate"
        ] = safety_factor * coolant_mass_flow_rate / density

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pump_id = self.options["pump_id"]

        density = inputs["coolant_density"]
        coolant_mass_flow_rate = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate"
        ]
        safety_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":safety_factor"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate",
            "coolant_density",
        ] = -safety_factor * coolant_mass_flow_rate / density**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":coolant:mass_flow_rate",
        ] = safety_factor / density

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":volumetric_flow_rate",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":safety_factor",
        ] = coolant_mass_flow_rate / density

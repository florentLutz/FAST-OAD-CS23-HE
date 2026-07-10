# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingNozzleExitArea(om.ExplicitComponent):
    """
    Computation of the outlet area of the nozzle.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="connected_air_inlet_id",
            default=None,
            desc="Identifier of the connected air flush_inlet",
            allow_none=False,
        )
        self.options.declare(
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_flow_area",
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor",
            units="unitless",
            val=3.0,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
            val=0.033,
            units="m**2",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]
        nozzle_id = self.options["nozzle_id"]

        mass_flow_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor"
        ]
        design_flow_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_flow_area"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area"
        ] = design_flow_area * (mass_flow_factor - 1.0) / mass_flow_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_air_inlet_id = self.options["connected_air_inlet_id"]
        nozzle_id = self.options["nozzle_id"]

        mass_flow_factor = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor"
        ]
        design_flow_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_flow_area"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":design_flow_area",
        ] = (mass_flow_factor - 1.0) / mass_flow_factor

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_air_inlet_id
            + ":mass_flow_factor",
        ] = design_flow_area / mass_flow_factor**2.0

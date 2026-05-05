# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingNozzleAreaRatio(om.ExplicitComponent):
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
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
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
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":no_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":coolant_flow_length",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
            val=np.nan,
            units="m**2",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            units="unitless",
            val=2.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        nozzle_id = self.options["nozzle_id"]

        exit_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area"
        ]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":no_flow_length"
        ]
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":coolant_flow_length"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio"
        ] = (no_flow_length * coolant_flow_length) / exit_area

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        connected_heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        nozzle_id = self.options["nozzle_id"]

        exit_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area"
        ]
        no_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":no_flow_length"
        ]
        coolant_flow_length = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":coolant_flow_length"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
        ] = -(no_flow_length * coolant_flow_length) / exit_area**2.0

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":no_flow_length",
        ] = coolant_flow_length / exit_area

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + connected_heat_exchanger_id
            + ":coolant_flow_length",
        ] = no_flow_length / exit_area

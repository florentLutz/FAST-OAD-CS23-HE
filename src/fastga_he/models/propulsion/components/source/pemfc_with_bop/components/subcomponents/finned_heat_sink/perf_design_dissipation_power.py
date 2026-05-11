# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDesignDissipationPower(om.ExplicitComponent):
    """
    Computation of the design dissipation power of the finned heat sink.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max",
            units="kW",
            val=np.nan,
            desc="Maximum total thermal power of the PEMFC stack has to provide during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power of the PEMFC stack has to provide during the mission",
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat",
            units="kW",
            val=np.nan,
            desc="Maximum heat to dissipate from the compressed air.",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power",
            units="kW",
            val=100.0,
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials("*", "*", val=1.0)
        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max",
            -1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        thermal_power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":thermal_power_max"
        ]
        power_max = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":power_max"
        ]
        max_compressed_air_heat = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":max_compressed_air_heat"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + finned_heat_sink_id
            + ":design_dissipation_power"
        ] = thermal_power_max - power_max + max_compressed_air_heat

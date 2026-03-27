# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleAirSpeed(om.ExplicitComponent):
    """
    Computation of the average and exit air speed of the nozzle.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "entry_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            units="unitless",
            val=2.0,
        )

        self.add_output("average_air_speed", val=100, units="m/s", shape=number_of_points)
        self.add_output("exit_air_speed", val=140, units="m/s", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.declare_partials(
            of="*",
            wrt="entry_air_speed",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        outputs["average_air_speed"] = (
            inputs["entry_air_speed"]
            * (
                inputs[
                    "data:propulsion:he_power_train:PEMFC_stack_bop:"
                    + pemfc_stack_bop_id
                    + ":"
                    + nozzle_id
                    + ":area_ratio"
                ]
                + 1.0
            )
            / 2.0
        )
        outputs["exit_air_speed"] = (
            inputs["entry_air_speed"]
            * inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + nozzle_id
                + ":area_ratio"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        partials["average_air_speed", "entry_air_speed"] = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + nozzle_id
                + ":area_ratio"
            ]
            + 1.0
        ) / 2.0

        partials[
            "average_air_speed",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
        ] = inputs["entry_air_speed"] / 2.0

        partials["exit_air_speed", "entry_air_speed"] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio"
        ]

        partials[
            "exit_air_speed",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":area_ratio",
        ] = inputs["entry_air_speed"]

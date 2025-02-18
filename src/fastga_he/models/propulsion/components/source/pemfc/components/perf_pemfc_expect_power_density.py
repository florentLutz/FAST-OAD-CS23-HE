# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

MAX_PEMFC_SYSYEM_POWER_DENSITY = 500  # [kW/m^3]
MAX_PEMFC_STACK_POWER_DENSITY = 6000  # [kW/m^3]


class PerformancesPEMFCMaxPowerDensityFuelCellSystem(om.ExplicitComponent):
    """
    Computation of the maximum power density of PEMFC system excluding the inlet compressor.
    Applied in volume calculation.
    Source: https://www.h3dynamics.com/_files/ugd/3029f7_5111f6ea97244ed09b72a916a8997773.pdf
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power to the pemfc during the mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":max_power_density",
            units="kW/m**3",
            val=250.0,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        power_max = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_power_density = 19.816 * np.log(power_max) + 236.48

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_power_density"
        ] = np.clip(unclipped_power_density, 230.0, MAX_PEMFC_SYSYEM_POWER_DENSITY)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_power_density = 19.816 * np.log(power_max) + 236.48
        if (
            unclipped_power_density <= MAX_PEMFC_SYSYEM_POWER_DENSITY
            and unclipped_power_density >= 230.0
        ):
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 19.816 / power_max
        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0


class PerformancesPEMFCMaxPowerDensityFuelCellStack(om.ExplicitComponent):
    """
    Computation of the maximum power density of PEMFC stack excluding all BoPs. Applied in volume
    calculation.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power to the pemfc during the mission",
        )

        self.add_output(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":max_power_density",
            units="kW/m**3",
            val=1000.0,
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        power_max = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_power_density = 85.474 * power_max + 425.31

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_power_density"
        ] = np.clip(unclipped_power_density, 400.0, MAX_PEMFC_STACK_POWER_DENSITY)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        unclipped_power_density = (
            85.474
            * inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"]
            + 425.31
        )

        if (
            unclipped_power_density <= MAX_PEMFC_STACK_POWER_DENSITY
            and unclipped_power_density >= 400.0
        ):
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = (
                85.474
                / inputs[
                    "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
                ]
            )
        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0

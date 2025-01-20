# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad
from ..constants import SUBMODEL_PERFORMANCES_PEMFC_MAX_POWER_DENSITY

MAX_PEMFC_POWER_DENSITY = 2.06  # kW/kg

oad.RegisterSubmodel.active_models[
    SUBMODEL_PERFORMANCES_PEMFC_MAX_POWER_DENSITY
] = "fastga_he.submodel.propulsion.performances.pemfc.max_power_density.aerostak"


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_MAX_POWER_DENSITY,
    "fastga_he.submodel.propulsion.performances.pemfc.max_power_density.aerostak",
)
class PerformancesPEMFCMaxPowerDensityAerostak(om.ExplicitComponent):
    """
    Computation of the max power provide per kilogram of pemfc. Applied in weight calculation
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
            units="kW/kg",
            val=0.3,
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

        unclipped_power_density = 0.0344 * np.log(power_max) + 0.4564

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_power_density"
        ] = np.clip(unclipped_power_density, 0.05, MAX_PEMFC_POWER_DENSITY)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        power_max = inputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
        ]

        unclipped_power_density = 0.0344 * np.log(power_max) + 0.4564
        if unclipped_power_density <= MAX_PEMFC_POWER_DENSITY and unclipped_power_density >= 0.05:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = (
                0.0344 / power_max
            )
        else:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = 0.0


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_MAX_POWER_DENSITY,
    "fastga_he.submodel.propulsion.performances.pemfc.max_power_density.intelligent_energy",
)
class PerformancesPEMFCMaxPowerDensityIntelligentEnergy(om.ExplicitComponent):
    """
    Computation of the max power provide per kilogram of pemfc. Applied in weight calculation
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
            units="kW/kg",
            val=1.0,
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

        unclipped_power_density = 0.27775 * np.log(power_max) + 1.598

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":max_power_density"
        ] = np.clip(unclipped_power_density, 0.05, MAX_PEMFC_POWER_DENSITY)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        unclipped_power_density = (
            0.27775
            * np.log(
                inputs[
                    "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max"
                ]
            )
            + 1.598
        )

        if unclipped_power_density <= MAX_PEMFC_POWER_DENSITY and unclipped_power_density >= 0.05:
            partials[
                "data:propulsion:he_power_train:pemfc_stack:"
                + pemfc_stack_id
                + ":max_power_density",
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":power_max",
            ] = (
                0.27775
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

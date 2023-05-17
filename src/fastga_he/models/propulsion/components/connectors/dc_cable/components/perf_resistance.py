# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE

oad.RegisterSubmodel.active_models[
    SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE
] = "fastga_he.submodel.propulsion.performances.dc_line.resistance_profile.from_temperature"


@oad.RegisterSubmodel(
    SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE,
    "fastga_he.submodel.propulsion.performances.dc_line.resistance_profile.from_temperature",
)
class PerformancesResistance(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "cable_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
            val=np.nan,
            units="degK**-1",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature",
            val=293.15,
            units="degK",
        )

        self.add_output(
            "resistance_per_cable",
            val=np.full(number_of_points, 1.0e-3),
            units="ohm",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        cable_temperature = inputs["cable_temperature"]
        alpha_r = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor"
        ]
        reference_resistance = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance"
        ]
        reference_temperature = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature"
        ]

        resistance = reference_resistance * (
            1.0 + alpha_r * (cable_temperature - reference_temperature)
        )

        outputs["resistance_per_cable"] = resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        cable_temperature = inputs["cable_temperature"]
        alpha_r = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor"
        ]
        reference_resistance = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance"
        ]
        reference_temperature = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature"
        ]

        partials["resistance_per_cable", "cable_temperature"] = np.diag(
            np.full_like(cable_temperature, reference_resistance * alpha_r)
        )
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
        ] = reference_resistance * (cable_temperature - reference_temperature)
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance",
        ] = 1.0 + alpha_r * (cable_temperature - reference_temperature)
        partials[
            "resistance_per_cable",
            "settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature",
        ] = -np.full_like(cable_temperature, reference_resistance * alpha_r)

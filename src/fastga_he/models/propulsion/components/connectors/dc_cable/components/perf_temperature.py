# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE

SUBMODEL_DC_LINE_TEMPERATURE_STEADY_STATE = (
    "fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.steady_state"
)

oad.RegisterSubmodel.active_models[
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE
] = SUBMODEL_DC_LINE_TEMPERATURE_STEADY_STATE


@oad.RegisterSubmodel(
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
    SUBMODEL_DC_LINE_TEMPERATURE_STEADY_STATE,
)
class PerformancesTemperature(om.ExplicitComponent):
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
            "conduction_losses",
            units="W",
            desc="Joule losses in one cable of the harness",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            "heat_transfer_coefficient",
            val=np.full(number_of_points, 50.0),
            units="W/m**2/degK",
            desc="Heat transfer coefficient between cable and outside medium",
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature outside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            val=np.nan,
            units="m",
        )
        self.add_input("time_step", shape=number_of_points, units="s", val=np.nan)

        self.add_output(
            "cable_temperature",
            val=np.full(number_of_points, 288.15),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
                "exterior_temperature",
                "heat_transfer_coefficient",
                "conduction_losses",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        cable_radius = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ]
        cable_length = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"
        ]

        h = inputs["heat_transfer_coefficient"]

        temp_ext = inputs["exterior_temperature"]

        q_c = inputs["conduction_losses"]

        temperature_profile = temp_ext + q_c / (2.0 * np.pi * cable_radius * cable_length * h)

        outputs["cable_temperature"] = temperature_profile

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        cable_radius = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ]
        cable_length = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"
        ]

        h = inputs["heat_transfer_coefficient"]

        q_c = inputs["conduction_losses"]

        partials[
            "cable_temperature",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
        ] = -q_c / (2.0 * np.pi * cable_radius ** 2.0 * cable_length * h)
        partials[
            "cable_temperature",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = -q_c / (2.0 * np.pi * cable_radius * cable_length ** 2.0 * h)
        partials["cable_temperature", "heat_transfer_coefficient"] = np.diag(
            -q_c / (2.0 * np.pi * cable_radius * cable_length * h ** 2.0)
        )
        partials["cable_temperature", "conduction_losses"] = np.diag(
            1.0 / (2.0 * np.pi * cable_radius * cable_length * h ** 2.0)
        )
        partials["cable_temperature", "exterior_temperature"] = np.eye(number_of_points)

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCasingTemperature(om.ExplicitComponent):
    """
    Computation the temperature of one of the rectifier casing based on the losses on the previous
    point. The temperature can be calculated for one module and generalized to the two other
    since we assume they are identical. Assumes that the thermal constant is so small that we can
    consider that the losses equals the power we can dissipate in steady state.

    According to Semikron technical information, the heat transfer from junction to heat sink
    can be either modeled with a casing plate considered common to all junctions (consequently,
    a common R_th_cs for all modules), or consider individual R_th_cs. We will choose the former.

    Can be seen in :cite:`erroui:2019` and :cite:`tan:2022`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "heat_sink_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the heat sink",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":casing:thermal_resistance",
            units="K/W",
            val=np.nan,
            desc="Thermal resistance between the casing and the heat sink",
        )
        self.add_input(
            "losses_rectifier",
            units="W",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            "casing_temperature",
            val=np.full(number_of_points, 273.15),
            units="degK",
            desc="Temperature of the rectifier casing",
            shape=number_of_points,
        )

        self.declare_partials(
            of="casing_temperature",
            wrt="heat_sink_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )
        self.declare_partials(
            of="casing_temperature",
            wrt="losses_rectifier",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="casing_temperature",
            wrt="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":casing:thermal_resistance",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]
        temp_hs = inputs["heat_sink_temperature"]
        r_th_js = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":casing:thermal_resistance"
        ]

        # We assume that module dissipate their heat the same way, so we will only look at the
        # evolution in one module
        losses_one_module = inputs["losses_rectifier"] / 3.0

        outputs["casing_temperature"] = temp_hs + losses_one_module * r_th_js

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        r_th_js = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":casing:thermal_resistance"
        ]
        losses_one_module = inputs["losses_rectifier"] / 3.0

        partials["casing_temperature", "losses_rectifier"] = (
            r_th_js / 3.0 * np.ones(number_of_points)
        )
        partials[
            "casing_temperature",
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":casing:thermal_resistance",
        ] = losses_one_module

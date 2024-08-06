# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesShaftPowerIn(om.ExplicitComponent):
    """
    Component which computes the input shaft power based on the output shaft power and an assumed
    constant efficiency. Default value for the efficiency are taken from literature (see
    :cite:`thauvin:2018` and :cite:`pettes:2021`).
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_input(
            name="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":efficiency",
            val=0.98,
            desc="Efficiency of the planetary gear",
        )
        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)
        self.add_input(
            "power_split",
            units="percent",
            val=np.nan,
            shape=number_of_points,
            desc="Percent of the power going to the first (primary) input, in %, with a format "
            "adapted to mission",
        )

        self.add_output("shaft_power_in_1", units="kW", val=5000.0, shape=number_of_points)
        self.add_output("shaft_power_in_2", units="kW", val=5000.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt=["shaft_power_out", "power_split"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":efficiency",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        planetary_gear_id = self.options["planetary_gear_id"]

        power_out = inputs["shaft_power_out"]
        eta = inputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency"
        ]
        percent_split = inputs["power_split"] / 100

        outputs["shaft_power_in_1"] = power_out / eta * percent_split
        outputs["shaft_power_in_2"] = power_out / eta * (1.0 - percent_split)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        planetary_gear_id = self.options["planetary_gear_id"]

        power_out = inputs["shaft_power_out"]
        eta = inputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency"
        ]
        percent_split = inputs["power_split"] / 100

        partials["shaft_power_in_1", "shaft_power_out"] = percent_split / eta
        partials[
            "shaft_power_in_1",
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency",
        ] = -power_out / eta**2.0 * percent_split
        partials["shaft_power_in_1", "power_split"] = power_out / eta / 100.0

        partials["shaft_power_in_2", "shaft_power_out"] = (1.0 - percent_split) / eta
        partials[
            "shaft_power_in_2",
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":efficiency",
        ] = -power_out / eta**2.0 * (1.0 - percent_split)
        partials["shaft_power_in_2", "power_split"] = -power_out / eta / 100.0

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesShaftPowerIn(om.ExplicitComponent):
    """
    Component which computes the input shaft power based on the output shaft powers and an assumed
    constant efficiency. Default value for the efficiency are taken from literature (see
    :cite:`thauvin:2018` and :cite:`pettes:2021`).
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        gearbox_id = self.options["gearbox_id"]

        self.add_input(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":efficiency",
            val=0.98,
            desc="Efficiency of the planetary gear",
        )
        self.add_input("shaft_power_out_1", units="kW", val=np.nan, shape=number_of_points)
        self.add_input("shaft_power_out_2", units="kW", val=np.nan, shape=number_of_points)

        self.add_output("shaft_power_in", units="kW", val=5000.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt=["shaft_power_out_1", "shaft_power_out_2"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":efficiency"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gearbox_id = self.options["gearbox_id"]

        power_out = inputs["shaft_power_out_1"] + inputs["shaft_power_out_2"]
        eta = inputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":efficiency"]

        outputs["shaft_power_in"] = power_out / eta

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gearbox_id = self.options["gearbox_id"]
        number_of_points = self.options["number_of_points"]

        power_out = inputs["shaft_power_out_1"] + inputs["shaft_power_out_2"]
        eta = inputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":efficiency"]

        partials["shaft_power_in", "shaft_power_out_1"] = np.ones(number_of_points) / eta
        partials["shaft_power_in", "shaft_power_out_2"] = np.ones(number_of_points) / eta
        partials[
            "shaft_power_in",
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":efficiency",
        ] = -power_out / eta**2.0

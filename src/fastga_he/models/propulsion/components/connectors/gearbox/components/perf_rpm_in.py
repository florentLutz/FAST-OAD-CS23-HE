# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesRPMIn(om.ExplicitComponent):
    """
    Component which computes the input rpms based on the output rpm and gear ratio. The rpm will
    be assumed to be equal on outputs so we'll only consider output number one.
    """

    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        gearbox_id = self.options["gearbox_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":gear_ratio",
            val=np.nan,
            desc="Gear ratio of the planetary gear",
        )
        self.add_input("rpm_out_1", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("rpm_in", units="min**-1", val=5000.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="rpm_out_1",
            method="exact",
            cols=np.arange(number_of_points),
            rows=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":gear_ratio",
            method="exact",
            cols=np.zeros(number_of_points),
            rows=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gearbox_id = self.options["gearbox_id"]

        input_rpm = (
            inputs["rpm_out_1"]
            * inputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":gear_ratio"]
        )

        outputs["rpm_in"] = input_rpm

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gearbox_id = self.options["gearbox_id"]
        number_of_points = self.options["number_of_points"]

        partials["rpm_in", "rpm_out_1"] = (
            np.ones(number_of_points)
            * inputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":gear_ratio"]
        )
        partials[
            "rpm_in",
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":gear_ratio",
        ] = inputs["rpm_out_1"]

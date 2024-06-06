# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesRPMIn(om.ExplicitComponent):
    """
    Component which computes the input rpms based on the output rpm and gear ratio. The rpm will
    be assumed to be equal on both inputs.
    """

    def initialize(self):

        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        planetary_gear_id = self.options["planetary_gear_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":gear_ratio",
            val=np.nan,
            desc="Gear ratio of the planetary gear",
        )
        self.add_input("rpm_out", units="min**-1", val=np.nan, shape=number_of_points)

        # Choice was made to start input rpm numbering at 1 to irritate any future programmer
        # working on this code
        self.add_output("rpm_in_1", units="min**-1", val=5000.0, shape=number_of_points)
        self.add_output("rpm_in_2", units="min**-1", val=5000.0, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt="rpm_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":gear_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        input_rpm = (
            inputs["rpm_out"]
            * inputs[
                "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":gear_ratio"
            ]
        )

        outputs["rpm_in_1"] = input_rpm
        outputs["rpm_in_2"] = input_rpm

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]
        number_of_points = self.options["number_of_points"]

        # For compactness
        for input_number in ("1", "2"):
            partials["rpm_in_" + input_number, "rpm_out"] = (
                np.ones(number_of_points)
                * inputs[
                    "data:propulsion:he_power_train:planetary_gear:"
                    + planetary_gear_id
                    + ":gear_ratio"
                ]
            )
            partials[
                "rpm_in_" + input_number,
                "data:propulsion:he_power_train:planetary_gear:"
                + planetary_gear_id
                + ":gear_ratio",
            ] = inputs["rpm_out"]

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum torque, rpm, tip mach and advance ratio of the propeller.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("tip_mach", val=np.full(number_of_points, np.nan))
        self.add_input("advance_ratio", val=np.full(number_of_points, np.nan))
        self.add_input("torque_in", units="N*m", val=np.full(number_of_points, np.nan))
        self.add_input("rpm", units="min**-1", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":tip_mach_max",
            val=0.8,
            desc="Maximum value of the propeller tip mach Number",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:" + propeller_id + ":tip_mach_max",
            wrt="tip_mach",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":advance_ratio_max",
            val=1.5,
            desc="Maximum value of the propeller tip mach Number",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:" + propeller_id + ":advance_ratio_max",
            wrt="advance_ratio",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max",
            units="N*m",
            val=5000.0,
            desc="Maximum value of the propeller torque",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max",
            wrt="torque_in",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max",
            units="min**-1",
            val=5000.0,
            desc="Maximum value of the propeller rpm",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max",
            wrt="rpm",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":tip_mach_max"] = (
            np.max(inputs["tip_mach"])
        )
        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":advance_ratio_max"
        ] = np.max(inputs["advance_ratio"])
        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max"] = (
            np.max(inputs["torque_in"])
        )
        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max"] = np.max(
            inputs["rpm"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":tip_mach_max", "tip_mach"
        ] = np.where(inputs["tip_mach"] == np.max(inputs["tip_mach"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":advance_ratio_max",
            "advance_ratio",
        ] = np.where(inputs["advance_ratio"] == np.max(inputs["advance_ratio"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_max", "torque_in"
        ] = np.where(inputs["torque_in"] == np.max(inputs["torque_in"]), 1.0, 0.0)
        partials["data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_max", "rpm"] = (
            np.where(inputs["rpm"] == np.max(inputs["rpm"]), 1.0, 0.0)
        )

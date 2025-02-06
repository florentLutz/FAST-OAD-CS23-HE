# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesPEMFCPowerDensity(om.ExplicitComponent):
    """
    Computation of the power provide per kilogram of pemfc. As of when I wrote this, it will only
    be used as a post-processing value
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            units="kg",
            val=1000.0,
        )

        self.add_input("power_out", units="kW", val=np.full(number_of_points, np.nan))

        self.add_output("power_density", units="kW/kg", val=np.full(number_of_points, 5.0))

        self.declare_partials(
            of="*",
            wrt="power_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["power_density"] = (
            inputs["power_out"]
            / inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials["power_density", "power_out"] = (
            np.ones(number_of_points)
            / inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass"]
        )
        partials[
            "power_density",
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass",
        ] = (
            -inputs["power_out"]
            / inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":mass"] ** 2
        )

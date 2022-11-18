# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

C0_COPPER = 52.13
C1_COPPER = -85.71
C2_COPPER = 75.30

C0_ALU = 42.62
C1_ALU = -70.68
C2_ALU = 60.33


class SizingCableGauge(om.Group):
    """Computation of the cable gauge."""

    def initialize(self):

        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        self.add_subsystem(
            "discriminant", _Discriminant(harness_id=harness_id), promotes=["data:*"]
        )
        self.add_subsystem("log_solution", _LogSolution(), promotes=[])
        self.add_subsystem(
            "actual_solution", _SolutionGauge(harness_id=harness_id), promotes=["data:*"]
        )

        self.connect("discriminant.discriminant", "log_solution.discriminant")
        self.connect("discriminant.c_1", "log_solution.c_1")
        self.connect("discriminant.c_2", "log_solution.c_2")

        self.connect("log_solution.log_solution", "actual_solution.log_solution")


class _Discriminant(om.ExplicitComponent):
    """Computation the discriminant to find the proper cable gauge."""

    def initialize(self):

        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current",
            val=np.nan,
            units="A",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            val=1.0,
            desc="1.0 for copper, 0.0 for aluminium",
        )

        self.add_output(name="discriminant")
        self.add_output(name="c_1")
        self.add_output(name="c_2")

        self.declare_partials(of="discriminant", wrt="*")
        self.declare_partials(
            of="c_1",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        )
        self.declare_partials(
            of="c_2",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        current = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current"
        ]
        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        c_0 = C0_ALU + (C0_COPPER - C0_ALU) * material - current
        c_1 = C1_ALU + (C1_COPPER - C1_ALU) * material
        c_2 = C2_ALU + (C2_COPPER - C2_ALU) * material

        delta = c_1 ** 2.0 - 4.0 * c_0 * c_2

        outputs["discriminant"] = delta
        outputs["c_1"] = c_1
        outputs["c_2"] = c_2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        current = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current"
        ]
        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        c_0 = C0_ALU + (C0_COPPER - C0_ALU) * material - current
        c_1 = C1_ALU + (C1_COPPER - C1_ALU) * material
        c_2 = C2_ALU + (C2_COPPER - C2_ALU) * material

        d_c_0_d_mat = C0_COPPER - C0_ALU
        d_c_0_d_current = -1.0
        d_c_1_d_mat = C1_COPPER - C1_ALU
        d_c_2_d_mat = C2_COPPER - C2_ALU

        partials[
            "discriminant",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:current",
        ] = (
            -4.0 * c_2 * d_c_0_d_current
        )

        partials[
            "discriminant",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = 2.0 * c_1 * d_c_1_d_mat - 4.0 * (c_0 * d_c_2_d_mat + c_2 * d_c_0_d_mat)

        partials[
            "c_1",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = d_c_1_d_mat
        partials[
            "c_2",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = d_c_2_d_mat


class _LogSolution(om.ExplicitComponent):
    """Computation the log of the solution to find the cable gauge."""

    def setup(self):

        self.add_input(name="discriminant", val=np.nan)
        self.add_input(name="c_1", val=np.nan)
        self.add_input(name="c_2", val=np.nan)

        self.add_output("log_solution")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        log_solution = (-inputs["c_1"] + np.sqrt(inputs["discriminant"])) / (2.0 * inputs["c_2"])

        outputs["log_solution"] = log_solution

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["log_solution", "discriminant"] = 0.5 / (
            2.0 * inputs["c_2"] * np.sqrt(inputs["discriminant"])
        )
        partials["log_solution", "c_1"] = -1.0 / (2.0 * inputs["c_2"])
        partials["log_solution", "c_2"] = -(-inputs["c_1"] + np.sqrt(inputs["discriminant"])) / (
            2.0 * inputs["c_2"] ** 2.0
        )


class _SolutionGauge(om.ExplicitComponent):
    """Computation the of the area to find the cable gauge based on its log."""

    def initialize(self):

        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        self.add_input("log_solution")
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:k_factor_radius",
            val=1.0,
            desc="K-factor to increase the radius of the conducting core",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:section",
            units="mm*mm",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:radius",
            units="mm",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        section = np.exp(inputs["log_solution"])
        k_factor = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:k_factor_radius"
        ]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section"
        ] = (section * k_factor ** 2.0)

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius"
        ] = (np.sqrt(section / np.pi) * k_factor)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        section = np.exp(inputs["log_solution"])
        k_factor = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:k_factor_radius"
        ]

        d_section_d_log_solution = np.exp(inputs["log_solution"])

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section",
            "log_solution",
        ] = (
            d_section_d_log_solution * k_factor ** 2.0
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:section",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:k_factor_radius",
        ] = (
            section * 2.0 * k_factor
        )

        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
            "log_solution",
        ] = (
            0.5 * np.sqrt(1.0 / (section * np.pi)) * d_section_d_log_solution * k_factor
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:k_factor_radius",
        ] = np.sqrt(section / np.pi)

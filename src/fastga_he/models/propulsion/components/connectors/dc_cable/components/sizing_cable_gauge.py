# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import numpy as np
import openmdao.api as om

C0_COPPER = 1.23553482e-01
C1_COPPER = 4.44191525e-02
C2_COPPER = -1.88835315e-04
C3_COPPER = 4.48406500e-07
C4_COPPER = -5.24550340e-10
C5_COPPER = 2.38160582e-13

C0_ALU = -2.75147916e-02
C1_ALU = 5.87923841e-02
C2_ALU = -3.08804274e-04
C3_ALU = 8.75318539e-07
C4_ALU = -1.20500903e-09
C5_ALU = 6.39275239e-13

LIMIT_CU = 750
LIMIT_ALU = 630

_LOGGER = logging.getLogger(__name__)


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

        self.add_subsystem("log_solution", _LogSolution(harness_id=harness_id), promotes=["data:*"])
        self.add_subsystem(
            "actual_solution", _SolutionGauge(harness_id=harness_id), promotes=["data:*"]
        )

        self.connect("log_solution.log_solution", "actual_solution.log_solution")


class _LogSolution(om.ExplicitComponent):
    """Computation the log of the solution to find the cable gauge."""

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
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber",
            val=np.nan,
            units="A",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
            val=1.0,
            desc="1.0 for copper, 0.0 for aluminium",
        )

        self.add_output("log_solution")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        current = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber"
        ]
        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        limit_current = LIMIT_ALU + (LIMIT_CU - LIMIT_ALU) * material
        if current > limit_current:
            _LOGGER.warning(
                "Current in the cable greater than what the NEC table allow, "
                "consider using multiple cables in the harness"
            )
            current = limit_current

        c_0 = C0_ALU + (C0_COPPER - C0_ALU) * material
        c_1 = C1_ALU + (C1_COPPER - C1_ALU) * material
        c_2 = C2_ALU + (C2_COPPER - C2_ALU) * material
        c_3 = C3_ALU + (C3_COPPER - C3_ALU) * material
        c_4 = C4_ALU + (C4_COPPER - C4_ALU) * material
        c_5 = C5_ALU + (C5_COPPER - C5_ALU) * material

        log_solution = (
            c_5 * current ** 5.0
            + c_4 * current ** 4.0
            + c_3 * current ** 3.0
            + c_2 * current ** 2.0
            + c_1 * current
            + c_0
        )

        outputs["log_solution"] = log_solution

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        current = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber"
        ]
        material = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material"
        ]

        c_1 = C1_ALU + (C1_COPPER - C1_ALU) * material
        c_2 = C2_ALU + (C2_COPPER - C2_ALU) * material
        c_3 = C3_ALU + (C3_COPPER - C3_ALU) * material
        c_4 = C4_ALU + (C4_COPPER - C4_ALU) * material
        c_5 = C5_ALU + (C5_COPPER - C5_ALU) * material

        partials[
            "log_solution",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:current_caliber",
        ] = (
            5.0 * c_5 * current ** 4.0
            + 4.0 * c_4 * current ** 3.0
            + 3.0 * c_3 * current ** 2.0
            + 2.0 * c_2 * current
            + c_1
        )
        partials[
            "log_solution",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":material",
        ] = (
            (C5_COPPER - C5_ALU) * current ** 5.0
            + (C4_COPPER - C4_ALU) * current ** 4.0
            + (C3_COPPER - C3_ALU) * current ** 3.0
            + (C2_COPPER - C2_ALU) * current ** 2.0
            + (C1_COPPER - C1_ALU) * current
            + (C0_COPPER - C0_ALU)
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
            val=10.0,
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:radius",
            units="mm",
            val=2.0,
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

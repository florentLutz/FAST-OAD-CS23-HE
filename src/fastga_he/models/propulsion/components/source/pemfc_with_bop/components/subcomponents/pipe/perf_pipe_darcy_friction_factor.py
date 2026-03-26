# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPipeDarcyFrictionFactor(om.Group):
    """
    Maximum pressure drop computation of the humidifier during mission.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        self.add_subsystem(
            name="log10_term1",
            subsys=_Log10Term1(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="log10_term2",
            subsys=_Log10Term2(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["*"],
        )
        self.add_subsystem(
            name="log10",
            subsys=_Log10(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="darcy_friction_factor",
            subsys=_PipeDarcyFrictionFactor(pemfc_stack_bop_id=pemfc_stack_bop_id, pipe_id=pipe_id),
            promotes=["*"],
        )


class _Log10Term1(om.ExplicitComponent):
    """
    Term 1 inside the log function
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":material_absolute_roughness",
            units="m",
            val=1.5e-6,
            desc="Absolute roughness of the pipe material, default is the absolute roughness of "
            "copper",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="log10_term1",
            units="unitless",
            val=0.001,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        roughness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":material_absolute_roughness"
        ]
        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]

        outputs["log10_term1"] = (roughness / (7.4 * radius)) ** 1.11

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        roughness = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":material_absolute_roughness"
        ]
        radius = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius"
        ]

        partials[
            "log10_term1",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":material_absolute_roughness",
        ] = 1.11 * (roughness / (7.4 * radius)) ** 1.11 / roughness

        partials[
            "log10_term1",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":radius",
        ] = -1.11 * (roughness / (7.4 * radius)) ** 1.11 / radius


class _Log10Term2(om.ExplicitComponent):
    """
    Term 2 inside the log function
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            units="unitless",
            val=np.nan,
            desc="Reynolds number of the flow in the pipe",
        )

        self.add_output(
            name="log10_term2",
            units="unitless",
            val=0.001,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        outputs["log10_term2"] = (
            6.9
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":reynolds_number"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        partials[
            "log10_term2",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
        ] = (
            -6.9
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pipe_id
                + ":reynolds_number"
            ]
            ** 2.0
        )


class _Log10(om.ExplicitComponent):
    """
    Logarithm term for the friction factor computation
    """

    def setup(self):
        self.add_input("log10_term1", units="unitless", val=np.nan)
        self.add_input("log10_term2", units="unitless", val=np.nan)

        self.add_output("log10", units="unitless", val=np.nan)

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["log10"] = np.log10(inputs["log10_term1"] + inputs["log10_term2"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        log10_term1 = inputs["log10_term1"]
        log10_term2 = inputs["log10_term2"]
        sum_terms = log10_term1 + log10_term2

        partials["log10", "log10_term1"] = 1.0 / (sum_terms * np.log(10))

        partials["log10", "log10_term2"] = 1.0 / (sum_terms * np.log(10))


class _PipeDarcyFrictionFactor(om.ExplicitComponent):
    """
    Final computation of the Darcy friction factor using the log10 term
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        self.add_input("log10", units="unitless", val=np.nan)
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
            units="unitless",
            val=np.nan,
            desc="Reynolds number of the flow in the pipe",
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor",
            units="unitless",
            val=0.001,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        log10 = inputs["log10"]
        reynolds_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number"
        ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor"
        ] = np.where(reynolds_number <= 2000.0, 64.0 / reynolds_number, (-1.8 * log10) ** -2.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        pipe_id = self.options["pipe_id"]

        log10 = inputs["log10"]
        reynolds_number = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number"
        ]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor",
            "log10",
        ] = np.where(
            reynolds_number <= 2000.0,
            0.0,
            -2.0 / (3.24 * log10**3.0),
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":darcy_friction_factor",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pipe_id
            + ":reynolds_number",
        ] = np.where(
            reynolds_number <= 2000.0,
            -64.0 / reynolds_number**2.0,
            0.0,
        )

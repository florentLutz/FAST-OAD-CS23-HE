# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGeneratorLossCoefficient(om.ExplicitComponent):
    """
    Computation of loss coefficients factor for cylindrical generator.

    Main losses considered in this model are :
    - Joules losses (alpha * T^2).
    - Hysteresis losses (beta * omega).
    - Eddy current losses (gamma * omega^2).
    """

    def initialize(self):
        # Reference generator : EMRAX 268

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )
        self.options.declare(
            "alpha_ref",
            default=0.016,
            desc="Joule loss coefficient for the reference generator",
        )
        self.options.declare(
            "beta_ref", default=6.64, desc="Hysteresis loss coefficient for the reference generator"
        )
        self.options.declare(
            "gamma_ref",
            default=0.015,
            desc="Eddy current loss coefficient for the reference generator",
        )

    def setup(self):

        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:alpha",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:beta",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:gamma",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:alpha",
            val=self.options["alpha_ref"],
            units="W/N**2/m**2",
        )
        self.add_output(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:beta",
            val=self.options["beta_ref"],
            units="W*s/rad",
        )
        self.add_output(
            name="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:gamma",
            val=self.options["gamma_ref"],
            units="W*s**2/rad**2",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:alpha",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:alpha",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:beta",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:beta",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:"
            + generator_id
            + ":loss_coefficient:gamma",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:gamma",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        generator_id = self.options["generator_id"]

        alpha_ref = self.options["alpha_ref"]
        beta_ref = self.options["beta_ref"]
        gamma_ref = self.options["gamma_ref"]

        alpha_scaling = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:alpha"
        ]
        beta_scaling = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:beta"
        ]
        gamma_scaling = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:gamma"
        ]

        alpha = alpha_ref * alpha_scaling
        beta = beta_ref * beta_scaling
        gamma = gamma_ref * gamma_scaling

        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:alpha"
        ] = alpha
        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:beta"
        ] = beta
        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:gamma"
        ] = gamma

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        generator_id = self.options["generator_id"]

        alpha_ref = self.options["alpha_ref"]
        beta_ref = self.options["beta_ref"]
        gamma_ref = self.options["gamma_ref"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:alpha",
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:alpha",
        ] = alpha_ref

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:beta",
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:beta",
        ] = beta_ref
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":loss_coefficient:gamma",
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:gamma",
        ] = gamma_ref

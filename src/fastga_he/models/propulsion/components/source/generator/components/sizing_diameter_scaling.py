# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGeneratorDiameterScaling(om.ExplicitComponent):
    """
    Computation of scaling factor for the diameter of the generator.

    Formula taken from :cite:`budinger:2012` or :cite:`thauvin:2018`.
    """

    def initialize(self):
        # Reference generator : EMRAX 268

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )
        self.options.declare(
            "rpm_max_ref",
            default=4500.0,
            desc="Max rotational speed of the reference generator in [rpm]",
        )

    def setup(self):

        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":rpm_rating",
            val=np.nan,
            units="min**-1",
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter",
            wrt="data:propulsion:he_power_train:generator:" + generator_id + ":rpm_rating",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        generator_id = self.options["generator_id"]

        rpm_max_ref = self.options["rpm_max_ref"]

        rpm_max = inputs["data:propulsion:he_power_train:generator:" + generator_id + ":rpm_rating"]

        rpm_peak_scaling = rpm_max / rpm_max_ref

        # Mechanical limit
        d_scaling = 1.0 / rpm_peak_scaling

        outputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter"
        ] = d_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        generator_id = self.options["generator_id"]

        rpm_max_ref = self.options["rpm_max_ref"]

        rpm_max = inputs["data:propulsion:he_power_train:generator:" + generator_id + ":rpm_rating"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":scaling:diameter",
            "data:propulsion:he_power_train:generator:" + generator_id + ":rpm_rating",
        ] = (
            -1.0 * rpm_max_ref / rpm_max ** 2.0
        )

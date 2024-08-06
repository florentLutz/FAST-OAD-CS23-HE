# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorEnergy(om.ExplicitComponent):
    """
    Computation of the maximum magnetic energy stored in the filter inductor, implementation of the
    formula from :cite:`wildi:2005`. Direct implementation, in the openMDAO format of
    :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":inductor:current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the DC/DC converter",
        )
        self.add_input(
            name=prefix + ":inductor:inductance",
            units="H",
            val=np.nan,
            desc="Inductance of the inductor",
        )

        self.add_output(
            name=prefix + ":inductor:magnetic_energy_rating",
            units="J",
            val=50,
            desc="Maximum magnetic energy that can be stored in the inductors",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        current_caliber = inputs[prefix + ":inductor:current_caliber"]
        inductance = inputs[prefix + ":inductor:inductance"]

        outputs[prefix + ":inductor:magnetic_energy_rating"] = (
            0.5 * inductance * current_caliber**2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        current_caliber = inputs[prefix + ":inductor:current_caliber"]
        inductance = inputs[prefix + ":inductor:inductance"]

        partials[
            prefix + ":inductor:magnetic_energy_rating",
            prefix + ":inductor:current_caliber",
        ] = inductance * current_caliber
        partials[
            prefix + ":inductor:magnetic_energy_rating",
            prefix + ":inductor:inductance",
        ] = 0.5 * current_caliber**2.0

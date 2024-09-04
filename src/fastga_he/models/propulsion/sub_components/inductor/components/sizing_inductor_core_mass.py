# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorCoreMass(om.ExplicitComponent):
    """
    Computation of the mass of a single E-core for the filter inductor, implementation in the
    openMDAO format of :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )
        self.options.declare(
            name="core_mass_ref",
            types=float,
            default=493e-3,
            desc="Core mass of the reference component [kg]",
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            prefix + ":inductor:scaling:core_mass",
            val=np.nan,
            desc="Scaling factor for the mass of the E-core",
        )

        self.add_output(
            prefix + ":inductor:core_mass",
            units="kg",
            val=self.options["core_mass_ref"],
            desc="Mass of the E-core in the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":inductor:core_mass"] = (
            inputs[prefix + ":inductor:scaling:core_mass"] * self.options["core_mass_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        partials[prefix + ":inductor:core_mass", prefix + ":inductor:scaling:core_mass"] = (
            self.options["core_mass_ref"]
        )

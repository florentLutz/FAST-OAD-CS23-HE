# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorCoreDimensions(om.ExplicitComponent):
    """
    Computation of the dimensions of the E-core for the filter inductor, implementation in the
    openMDAO format of :cite:`budinger_sizing_2023`. We will only compute the dimensions relevant
    for the computation of the copper mass but the procedure is the same for the rest of the
    dimensions. The name of the dimensions correspond to what can be seen in
    methodology/E-core_dimensions.PNG.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )
        self.options.declare(
            name="b_ref",
            types=float,
            default=73.15e-3,
            desc="B dimension of the reference component [m]",
        )
        self.options.declare(
            name="c_ref",
            types=float,
            default=27.5e-3,
            desc="C dimension of the reference component [m]",
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            prefix + ":inductor:scaling:core_dimension",
            val=np.nan,
            desc="Scaling factor for the mass of the E-core",
        )

        self.add_output(
            prefix + ":inductor:core_dimension:B",
            units="m",
            val=self.options["b_ref"],
            desc="B dimension of the E-core in the inductor",
        )
        self.add_output(
            prefix + ":inductor:core_dimension:C",
            units="m",
            val=self.options["c_ref"],
            desc="C dimension of the E-core in the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":inductor:core_dimension:B"] = (
            inputs[prefix + ":inductor:scaling:core_dimension"] * self.options["b_ref"]
        )

        outputs[prefix + ":inductor:core_dimension:C"] = (
            inputs[prefix + ":inductor:scaling:core_dimension"] * self.options["c_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prefix = self.options["prefix"]

        partials[
            prefix + ":inductor:core_dimension:B", prefix + ":inductor:scaling:core_dimension"
        ] = self.options["b_ref"]
        partials[
            prefix + ":inductor:core_dimension:C", prefix + ":inductor:scaling:core_dimension"
        ] = self.options["c_ref"]

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInductorCopperWireArea(om.ExplicitComponent):
    """
    Computation of the copper wire section area for the inductor, implementation of the formula from
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
            name=prefix + ":inductor:wire_current_density",
            units="A/m**2",
            val=5e6,
            desc="Maximum current density allowed in the copper wire",
        )

        self.add_output(
            name=prefix + ":inductor:wire_section_area",
            units="m**2",
            val=100e-6,
            desc="Section area of the copper wire inside the inductor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":inductor:wire_section_area"] = (
            inputs[prefix + ":inductor:current_caliber"]
            / inputs[prefix + ":inductor:wire_current_density"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        prefix = self.options["prefix"]

        partials[prefix + ":inductor:wire_section_area", prefix + ":inductor:current_caliber",] = (
            1.0 / inputs[prefix + ":inductor:wire_current_density"]
        )
        partials[
            prefix + ":inductor:wire_section_area",
            prefix + ":inductor:wire_current_density",
        ] = -(
            inputs[prefix + ":inductor:current_caliber"]
            / inputs[prefix + ":inductor:wire_current_density"] ** 2.0
        )

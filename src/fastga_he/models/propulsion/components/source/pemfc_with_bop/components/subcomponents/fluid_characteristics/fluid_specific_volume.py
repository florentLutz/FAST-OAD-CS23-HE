# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .fluid_density import FluidDensity


class FluidSpecificVolume(om.Group):
    """
    Fluid specific volume calculation for heat transfer models.
    """

    def initialize(self):
        self.options.declare(
            "fluid",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        fluid = self.options["fluid"]

        self.add_subsystem(
            "fluid_density_" + fluid,
            FluidDensity(fluid=fluid),
            promotes=["*"],
        )
        self.add_subsystem(
            "specific_volume_" + fluid,
            _SpecificVolume(),
            promotes=["*"],
        )


class _SpecificVolume(om.ExplicitComponent):
    """
    Fluid specific volume calculation for heat transfer models.
    """

    def setup(self):
        self.add_input("fluid_density", val=np.nan, units="kg/m**3")

        self.add_output("fluid_specific_volume", val=0.82, units="m**3/kg")

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["fluid_specific_volume"] = 1.0 / inputs["fluid_density"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["fluid_specific_volume", "fluid_density"] = -1.0 / inputs["fluid_density"] ** 2.0

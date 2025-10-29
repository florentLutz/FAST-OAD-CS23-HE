# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials
from ..constants import DEFAULT_DYNAMIC_VISCOSITY


class PerformancesAirDynamicViscosity(om.ExplicitComponent):
    """
    Computation of the dynamic viscosity of the ambient air, which varies based on the change of
    operating altitude of the SM PMSM.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", units="m", val=np.zeros(number_of_points))

        self.add_output(
            name="dynamic_viscosity",
            units="kg/m/s",
            val=np.full(number_of_points, DEFAULT_DYNAMIC_VISCOSITY),
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["dynamic_viscosity"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).dynamic_viscosity

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["dynamic_viscosity", "altitude"] = AtmosphereWithPartials(
            inputs["altitude"], altitude_in_feet=False
        ).partial_dynamic_viscosity_altitude

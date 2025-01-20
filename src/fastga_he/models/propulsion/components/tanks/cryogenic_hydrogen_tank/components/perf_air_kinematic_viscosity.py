# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from stdatm import AtmosphereWithPartials

DEFAULT_KINEMATIC_VISCOSITY = 1.5 * 10**-5


class PerformancesAirKinematicViscosity(om.ExplicitComponent):
    """
    Computation of the free stream temperature of the exterior surface of the tank
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("altitude", units="m", val=np.zeros(number_of_points))

        self.add_output(
            name="air_kinematic_viscosity",
            units="m**2/s",
            val=np.full(number_of_points, DEFAULT_KINEMATIC_VISCOSITY),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["air_kinematic_viscosity"] = AtmosphereWithPartials(
            inputs["altitude"]
        ).kinematic_viscosity

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["air_kinematic_viscosity", "altitude"] = AtmosphereWithPartials(
            inputs["altitude"]
        ).partial_kinematic_viscosity_altitude

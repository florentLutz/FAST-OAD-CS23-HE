# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

HYDROGEN_VAPORIZATION_LATENT_HEAT = 446592.0  # J/kg


class PerformancesHydrogenBoilOffMission(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen boil-off during the mission.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "heat_conduction",
            units="J/s",
            val=np.full(number_of_points, np.nan),
            desc="Hydrogen from this tank consumed at each time step",
        )

        self.add_input("time_step", units="s", val=np.full(number_of_points, np.nan))

        self.add_output(
            "hydrogen_boil_off_t",
            units="kg",
            val=np.linspace(0.25, 0.15, number_of_points),
            desc="Hydrogen boil-off in the tank at each time step",
        )

        self.declare_partials(
            of="hydrogen_boil_off_t",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["hydrogen_boil_off_t"] = (
            inputs["time_step"] * inputs["heat_conduction"] / HYDROGEN_VAPORIZATION_LATENT_HEAT
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["hydrogen_boil_off_t", "time_step"] = (
            inputs["heat_conduction"] / HYDROGEN_VAPORIZATION_LATENT_HEAT
        )

        partials["hydrogen_boil_off_t", "heat_conduction"] = (
            inputs["time_step"] / HYDROGEN_VAPORIZATION_LATENT_HEAT
        )

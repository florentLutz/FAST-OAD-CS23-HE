# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirGapFluxDensity(om.ExplicitComponent):
    """
    Computation of the Maximum air gap magnetic flux density of the mototr. The formula is obtained
    from equation (II-6) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="tangential_stress",
            units="Pa",
            desc="The surface tangential stress applied by electromagnetism",
            shape=number_of_points,
            val=np.nan,
        )
        self.add_input(
            name="surface_current_density",
            val=np.nan,
            units="A/m",
            shape=number_of_points,
            desc="The maximum surface current density of the winding conductor cable",
        )

        self.add_output(
            name="air_gap_flux_density",
            val=0.9,
            units="T",
            shape=number_of_points,
            desc="The magnetic flux density provided by the permanent magnets",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        unclipped_flux_density = (
            2.0 * inputs["tangential_stress"] / inputs["surface_current_density"]
        )

        outputs["air_gap_flux_density"] = np.clip(unclipped_flux_density, 0.5, 1.2)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        unclipped_flux_density = (
            2.0 * inputs["tangential_stress"] / inputs["surface_current_density"]
        )
        clipped_flux_density = np.clip(unclipped_flux_density, 0.5, 1.2)

        partials["air_gap_flux_density", "surface_current_density"] = np.where(
            unclipped_flux_density == clipped_flux_density,
            -2.0 * inputs["tangential_stress"] / inputs["surface_current_density"] ** 2.0,
            1e-6,
        )

        partials["air_gap_flux_density", "tangential_stress"] = np.where(
            unclipped_flux_density == clipped_flux_density,
            2.0 / inputs["surface_current_density"],
            1e-6,
        )

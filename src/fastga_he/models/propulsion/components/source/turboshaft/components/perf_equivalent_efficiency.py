#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.models.environmental_impacts.simple_energy_impact import (
    ENERGY_CONTENT_JET_FUEL,
    KWH_TO_MJ,
)  # In MJ/kg


class PerformancesEquivalentEfficiency(om.ExplicitComponent):
    """
    Computation of an equivalent efficiency for the turboshaft. It will be computed as
    1 / (sfc * mu) where mu is the gravimetric energy density of the fuel. It is derived as
    eta = P_shaft / P_chemical = P_shaft / (m_f * mu) = 1 / (sfc * mu) where m_f is the fuel mass
    flow.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "specific_fuel_consumption", units="kg/kW/h", val=np.nan, shape=number_of_points
        )

        self.add_output(
            "equivalent_efficiency",
            val=np.full(number_of_points, 0.3),
            shape=number_of_points,
            desc="Equivalent efficiency of the turboshaft. Indicative, the sfc should be preferred",
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
        outputs["equivalent_efficiency"] = 1.0 / (
            inputs["specific_fuel_consumption"] * ENERGY_CONTENT_JET_FUEL / KWH_TO_MJ
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["equivalent_efficiency", "specific_fuel_consumption"] = -1.0 / (
            inputs["specific_fuel_consumption"] ** 2.0 * ENERGY_CONTENT_JET_FUEL / KWH_TO_MJ
        )

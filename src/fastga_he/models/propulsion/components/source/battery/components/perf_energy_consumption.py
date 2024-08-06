# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesEnergyConsumption(om.ExplicitComponent):
    """
    Computation of the consumable and non-consumable energy consumed as required by the
    FAST-OAD-GA framework to be compatible with the mission vector. These will then be added
    through an assembler, but each source should have a component like this one.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("voltage_out", units="V", val=np.full(number_of_points, np.nan))
        self.add_input("time_step", units="h", val=np.full(number_of_points, np.nan))

        self.add_output(
            "non_consumable_energy_t",
            val=np.full(number_of_points, 0.0),
            desc="fuel consumed at each time step in the battery",
            units="W*h",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["non_consumable_energy_t"] = (
            inputs["dc_current_out"] * inputs["voltage_out"] * inputs["time_step"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["non_consumable_energy_t", "dc_current_out"] = (
            inputs["voltage_out"] * inputs["time_step"]
        )
        partials["non_consumable_energy_t", "voltage_out"] = (
            inputs["dc_current_out"] * inputs["time_step"]
        )
        partials["non_consumable_energy_t", "time_step"] = (
            inputs["dc_current_out"] * inputs["voltage_out"]
        )

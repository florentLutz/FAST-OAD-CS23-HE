# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualEnergyCost(om.ExplicitComponent):
    """
    Computation of the annual energy cost of the aircraft.
    """

    def setup(self):
        self.add_input(
            name="data:cost:electric_energy_cost",
            val=0.0,
            units="USD",
            desc="Electric energy cost for single flight mission",
        )
        self.add_input(
            name="data:cost:fuel_cost",
            val=0.0,
            units="USD",
            desc="Fuel cost for single flight mission",
        )
        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        self.add_output(
            name="data:cost:operation:annual_fuel_cost",
            val=1000.0,
            units="USD/yr",
        )
        self.add_output(
            name="data:cost:operation:annual_electricity_cost",
            val=1000.0,
            units="USD/yr",
        )

        self.declare_partials("*", "data:TLAR:flight_per_year", method="exact")
        self.declare_partials(
            of="data:cost:operation:annual_fuel_cost",
            wrt="data:cost:fuel_cost",
            method="exact",
        )
        self.declare_partials(
            of="data:cost:operation:annual_electricity_cost",
            wrt="data:cost:electric_energy_cost",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_fuel_cost"] = (
            inputs["data:TLAR:flight_per_year"] * inputs["data:cost:fuel_cost"]
        )
        outputs["data:cost:operation:annual_electricity_cost"] = (
            inputs["data:TLAR:flight_per_year"] * inputs["data:cost:electric_energy_cost"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:cost:operation:annual_electricity_cost",
            "data:cost:electric_energy_cost",
        ] = inputs["data:TLAR:flight_per_year"]

        partials["data:cost:operation:annual_fuel_cost", "data:cost:fuel_cost"] = inputs[
            "data:TLAR:flight_per_year"
        ]

        partials["data:cost:operation:annual_electricity_cost", "data:TLAR:flight_per_year"] = (
            inputs["data:cost:electric_energy_cost"]
        )

        partials["data:cost:operation:annual_fuel_cost", "data:TLAR:flight_per_year"] = inputs[
            "data:cost:fuel_cost"
        ]

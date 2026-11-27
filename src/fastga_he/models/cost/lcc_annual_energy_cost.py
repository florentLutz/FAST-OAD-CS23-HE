# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualEnergyCost(om.ExplicitComponent):
    """
    Computation of the annual energy cost of the aircraft.
    """

    def setup(self):
        self.add_input(
            name="data:cost:electricity_cost",
            val=0.0,
            units="USD",
            desc="Electric energy cost for single flight mission",
        )
        self.add_input(
            "data:cost:hydrocarbon_fuel_cost",
            units="USD",
            val=0.0,
            desc="Fossil Fuel cost for single flight mission",
        )
        self.add_input(
            "data:cost:hydrogen_fuel_cost",
            units="USD",
            val=0.0,
            desc="Hydrogen Fuel cost for single flight mission",
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
            name="data:cost:operation:annual_hydrocarbon_fuel_cost",
            val=1000.0,
            units="USD/yr",
        )
        self.add_output(
            name="data:cost:operation:annual_hydrogen_fuel_cost",
            val=1000.0,
            units="USD/yr",
        )
        self.add_output(
            name="data:cost:operation:annual_electricity_cost",
            val=1000.0,
            units="USD/yr",
        )

    def setup_partials(self):
        self.declare_partials("*", "data:TLAR:flight_per_year", method="exact")
        self.declare_partials(
            of="data:cost:operation:annual_fuel_cost",
            wrt=["data:cost:hydrocarbon_fuel_cost", "data:cost:hydrogen_fuel_cost"],
            method="exact",
        )
        self.declare_partials(
            of="data:cost:operation:annual_electricity_cost",
            wrt="data:cost:electricity_cost",
            method="exact",
        )
        self.declare_partials(
            of="data:cost:operation:annual_hydrocarbon_fuel_cost",
            wrt="data:cost:hydrocarbon_fuel_cost",
            method="exact",
        )
        self.declare_partials(
            of="data:cost:operation:annual_hydrogen_fuel_cost",
            wrt="data:cost:hydrogen_fuel_cost",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        hydrocarbon_fuel_cost = inputs["data:cost:hydrocarbon_fuel_cost"]
        hydrogen_fuel_cost = inputs["data:cost:hydrogen_fuel_cost"]
        electricity_cost = inputs["data:cost:electricity_cost"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]

        outputs["data:cost:operation:annual_fuel_cost"] = (
            hydrogen_fuel_cost + hydrocarbon_fuel_cost
        ) * flight_per_year

        outputs["data:cost:operation:annual_electricity_cost"] = electricity_cost * flight_per_year

        outputs["data:cost:operation:annual_hydrocarbon_fuel_cost"] = (
            hydrocarbon_fuel_cost * flight_per_year
        )

        outputs["data:cost:operation:annual_hydrogen_fuel_cost"] = (
            hydrogen_fuel_cost * flight_per_year
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        hydrocarbon_fuel_cost = inputs["data:cost:hydrocarbon_fuel_cost"]
        hydrogen_fuel_cost = inputs["data:cost:hydrogen_fuel_cost"]
        electricity_cost = inputs["data:cost:electricity_cost"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]

        partials[
            "data:cost:operation:annual_electricity_cost",
            "data:cost:electricity_cost",
        ] = flight_per_year

        partials["data:cost:operation:annual_fuel_cost", "data:cost:hydrocarbon_fuel_cost"] = (
            flight_per_year
        )

        partials["data:cost:operation:annual_fuel_cost", "data:cost:hydrogen_fuel_cost"] = (
            flight_per_year
        )

        partials[
            "data:cost:operation:annual_hydrocarbon_fuel_cost",
            "data:cost:hydrocarbon_fuel_cost",
        ] = flight_per_year

        partials[
            "data:cost:operation:annual_hydrogen_fuel_cost",
            "data:cost:hydrogen_fuel_cost",
        ] = flight_per_year

        partials["data:cost:operation:annual_fuel_cost", "data:TLAR:flight_per_year"] = (
            hydrocarbon_fuel_cost + hydrogen_fuel_cost
        )

        partials["data:cost:operation:annual_electricity_cost", "data:TLAR:flight_per_year"] = (
            electricity_cost
        )

        partials[
            "data:cost:operation:annual_hydrocarbon_fuel_cost", "data:TLAR:flight_per_year"
        ] = hydrocarbon_fuel_cost

        partials["data:cost:operation:annual_hydrogen_fuel_cost", "data:TLAR:flight_per_year"] = (
            hydrogen_fuel_cost
        )

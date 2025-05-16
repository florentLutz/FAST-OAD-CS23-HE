# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualAirportCost(om.ExplicitComponent):
    """
    Computation of the aircraft annual airport related costs.
    """

    def setup(self):
        self.add_input(
            "data:cost:operation:landing_cost",
            val=np.nan,
            units="USD",
            desc="Landing cost per operation of the aircraft",
        )
        self.add_input(
            "data:cost:operation:daily_parking_cost",
            val=np.nan,
            units="USD/d",
            desc="Daily parking cost of the aircraft",
        )
        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        self.add_output(
            "data:cost:operation:annual_airport_cost",
            val=6.0e3,
            units="USD/yr",
            desc="Annual airport related cost for the aircraft",
        )

        self.declare_partials(
            of="*",
            wrt=["data:cost:operation:landing_cost", "data:TLAR:flight_per_year"],
            method="exact",
        )
        self.declare_partials(of="*", wrt="data:cost:operation:daily_parking_cost", val=365.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_airport_cost"] = (
            365.0 * inputs["data:cost:operation:daily_parking_cost"]
            + inputs["data:TLAR:flight_per_year"] * inputs["data:cost:operation:landing_cost"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:operation:annual_airport_cost", "data:TLAR:flight_per_year"] = inputs[
            "data:cost:operation:landing_cost"
        ]

        partials["data:cost:operation:annual_airport_cost", "data:cost:operation:landing_cost"] = (
            inputs["data:TLAR:flight_per_year"]
        )

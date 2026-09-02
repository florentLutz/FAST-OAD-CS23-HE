# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalRevenuePerRPK(om.ExplicitComponent):
    """
    Computation of the revenue per Revenue Passenger Kilometers (RPK). The RPK is the number of
    paying passengers multiplied by the distance flown.This is to verify the ticket pricing against
    the statisticcl data of SNCF from
    https://www.quechoisir.org/actualite-tarifs-sncf-au-kilometre-2026-le-kilometre-de-tgv
    -toujours-plus-cher-n174790/ .
    """

    def setup(self):
        self.add_input(
            "data:cost:operation:annual_revenue_per_unit",
            val=np.nan,
            units="USD/yr",
            desc="Annual operational cost per unit of the aircraft",
        )
        self.add_input(
            "data:TLAR:flight_per_year",
            val=np.nan,
        )
        self.add_input(
            "data:TLAR:range",
            val=np.nan,
            units="km",
            desc="The range of the aircraft",
        )
        self.add_input(
            "data:cost:operation:passenger_per_flight",
            val=np.nan,
            desc="The number of passengers per flight",
        )

        self.add_output(
            "data:cost:operation:revenue_per_rpk",
            units="USD/km",
            val=0.23,
            desc="Revenue per Revenue Passenger Kilometer (RPK)",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        annual_revenue_per_unit = inputs["data:cost:operation:annual_revenue_per_unit"]
        number_of_passenger = inputs["data:cost:operation:passenger_per_flight"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]
        range_km = inputs["data:TLAR:range"]

        outputs["data:cost:operation:revenue_per_rpk"] = annual_revenue_per_unit / (
            number_of_passenger * flight_per_year * range_km
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        annual_revenue_per_unit = inputs["data:cost:operation:annual_revenue_per_unit"]
        number_of_passenger = inputs["data:cost:operation:passenger_per_flight"]
        flight_per_year = inputs["data:TLAR:flight_per_year"]
        range_km = inputs["data:TLAR:range"]

        common_denominator = (
            number_of_passenger * flight_per_year * range_km
        )

        partials[
            "data:cost:operation:revenue_per_rpk", "data:cost:operation:annual_revenue_per_unit"
        ] = 1.0 / common_denominator

        partials["data:cost:operation:revenue_per_rpk", "data:cost:operation:passenger_per_flight"] = (
            -annual_revenue_per_unit / (common_denominator * number_of_passenger)
        )

        partials["data:cost:operation:revenue_per_rpk", "data:TLAR:flight_per_year"] = (
            -annual_revenue_per_unit / (common_denominator * flight_per_year)
        )

        partials["data:cost:operation:revenue_per_rpk", "data:TLAR:range"] = (
            -annual_revenue_per_unit / (common_denominator * range_km)
        )

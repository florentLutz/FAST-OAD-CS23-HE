# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalAnnualRevenue(om.Group):
    """
    Group to compute the annual revenue of the aircraft operator.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )
        self.options.declare(
            name="trainer",
            default=False,
            types=bool,
            desc="True if the aircraft is used as a trainer",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]
        trainer = self.options["trainer"]

        self.add_subsystem(
            "annual_revenue",
            _OperationalAnnualRevenue(trainer=trainer),
            promotes=["*"],
        )
        self.add_subsystem(
            "airfare_gain_factor",
            _AirfareGainFactor(years_of_service=years_of_service),
            promotes=["*"],
        )
        self.add_subsystem(
            "annual_revenue_projection",
            _OperationalAnnualRevenueProjection(years_of_service=years_of_service),
            promotes=["*"],
        )


class _AirfareGainFactor(om.ExplicitComponent):
    """
    Computation of the airfare gain factor from the CAGR of the airfare index. The airfare index
    history is obtained from https://fred.stlouisfed.org/series/CP0733EZ19M086NEST.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            "data:cost:operation:airfare_compound_annual_gain_rate",
            val=0.0298,
            desc="The compound annual growth rate (CAGR) of the airfare index.",
        )

        self.add_output(
            "data:cost:operation:airfare_compound_annual_gain_factor",
            val=1.0,
            desc="The airfare gain factor over the service life of the aircraft.",
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_service = self.options["years_of_service"]

        outputs["data:cost:operation:airfare_compound_annual_gain_factor"] = (
            1.0 + inputs["data:cost:operation:airfare_compound_annual_gain_rate"]
        ) ** np.arange(years_of_service)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        partials[
            "data:cost:operation:airfare_compound_annual_gain_factor",
            "data:cost:operation:airfare_compound_annual_gain_rate",
        ] = np.arange(years_of_service) * (
            1.0 + inputs["data:cost:operation:airfare_compound_annual_gain_rate"]
        ) ** (np.arange(years_of_service) - 1)


class _OperationalAnnualRevenue(om.ExplicitComponent):
    """
    Computation of the annual revenue of the aircraft. The profit margin is derived from the
    operating margin in Industry Statistics of IATA :cite:`iata:2025`. The per hour trainer cost is
    derived from the pricing listed in
    https://flyawaysimulation.com/ask/answers/real-world-flight-training-cost-per-hour/.
    """

    def initialize(self):
        self.options.declare(
            name="trainer",
            default=False,
            types=bool,
            desc="True if the aircraft is used as a trainer",
        )

    def setup(self):
        trainer = self.options["trainer"]

        if trainer:
            self.add_input(
                "data:cost:operation:trainer_hourly_cost",
                val=182.2,
                units="USD/h",
                desc="Hourly operational cost of the trainer aircraft, 182.2 USD/hr without "
                "instructor, 256.6 USD/hr with instructor",
            )
            self.add_input(
                name="data:TLAR:flight_hours_per_year",
                val=283.2,
                units="h",
                desc="Expected number of hours flown per year",
            )

        else:
            self.add_input(
                "data:cost:operation:annual_cost_per_unit",
                val=np.nan,
                units="USD/yr",
                desc="Annual operational cost per unit of the aircraft",
            )
            self.add_input(
                "data:cost:operation:average_annual_loan_cost_without_balloon",
                val=np.nan,
                units="USD/yr",
            )
            self.add_input(
                "data:cost:operation:profit_margin",
                val=0.07,
                desc="Profit margin as a fraction of the annual revenue",
            )

        self.add_output("data:cost:operation:annual_revenue_per_unit", units="USD/yr", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        trainer = self.options["trainer"]

        if trainer:
            outputs["data:cost:operation:annual_revenue_per_unit"] = (
                inputs["data:cost:operation:trainer_hourly_cost"]
                * inputs["data:TLAR:flight_hours_per_year"]
            )

        else:
            outputs["data:cost:operation:annual_revenue_per_unit"] = (
                inputs["data:cost:operation:annual_cost_per_unit"]
                + inputs["data:cost:operation:average_annual_loan_cost_without_balloon"]
            ) / (1.0 - inputs["data:cost:operation:profit_margin"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        trainer = self.options["trainer"]

        if trainer:
            partials[
                "data:cost:operation:annual_revenue_per_unit",
                "data:cost:operation:trainer_hourly_cost",
            ] = inputs["data:TLAR:flight_hours_per_year"]
            partials[
                "data:cost:operation:annual_revenue_per_unit", "data:TLAR:flight_hours_per_year"
            ] = inputs["data:cost:operation:trainer_hourly_cost"]

        else:
            partials[
                "data:cost:operation:annual_revenue_per_unit",
                "data:cost:operation:annual_cost_per_unit",
            ] = 1.0 / (1.0 - inputs["data:cost:operation:profit_margin"])

            partials[
                "data:cost:operation:annual_revenue_per_unit",
                "data:cost:operation:average_annual_loan_cost_without_balloon",
            ] = 1.0 / (1.0 - inputs["data:cost:operation:profit_margin"])

            partials[
                "data:cost:operation:annual_revenue_per_unit", "data:cost:operation:profit_margin"
            ] = (
                inputs["data:cost:operation:annual_cost_per_unit"]
                + inputs["data:cost:operation:average_annual_loan_cost_without_balloon"]
            ) / ((1.0 - inputs["data:cost:operation:profit_margin"]) ** 2.0)


class _OperationalAnnualRevenueProjection(om.ExplicitComponent):
    """
    Computation of the annual revenue projection of the aircraft over its service life.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            "data:cost:operation:annual_revenue_per_unit",
            val=np.nan,
            units="USD/yr",
            desc="Annual operational revenue per unit of the aircraft",
        )
        self.add_input(
            "data:cost:operation:airfare_compound_annual_gain_factor",
            val=np.nan,
            desc="The airfare gain factor over the service life of the aircraft.",
            shape=years_of_service,
        )

        self.add_output(
            "data:cost:operation:annual_revenue_projection",
            units="USD/yr",
            val=0.0,
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            of="*",
            wrt="data:cost:operation:annual_revenue_per_unit",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )
        self.declare_partials(
            of="*",
            wrt="data:cost:operation:airfare_compound_annual_gain_factor",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.arange(years_of_service),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_revenue_projection"] = (
            inputs["data:cost:operation:annual_revenue_per_unit"]
            * inputs["data:cost:operation:airfare_compound_annual_gain_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        partials[
            "data:cost:operation:annual_revenue_projection",
            "data:cost:operation:annual_revenue_per_unit",
        ] = inputs["data:cost:operation:airfare_compound_annual_gain_factor"]

        partials[
            "data:cost:operation:annual_revenue_projection",
            "data:cost:operation:airfare_compound_annual_gain_factor",
        ] = inputs["data:cost:operation:annual_revenue_per_unit"] * np.ones(years_of_service)

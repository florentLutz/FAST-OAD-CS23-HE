# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .lcc_operational_annual_revenue import LCCOperationalAnnualRevenue
from .lcc_passenger_per_flight import LCCOperationalPassengerPerFlight
from .lcc_operational_revenue_per_rpk import LCCOperationalRevenuePerRPK
from .lcc_annual_energy_cost_projection import LCCOperationalAnnualEnergyCostProjection
from .lcc_operation_annual_cash_flow import LCCOperationAnnualCashFlow
from .lcc_npv_discount_factor import LCCNPVDiscountFactor
from .lcc_net_present_value import LCCNetPresentValue
from .lcc_profidibility_index import LCCProfitabilityIndex
from .lcc_calculate_npax_design import LCCOperationalCalculateNPAXDesign


class LCCOperationalProfitability(om.Group):
    """
    Group collects all the Net Present Value (NPV) projection calculation.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )
        self.options.declare(
            name="loan",
            default=True,
            types=bool,
            desc="True if loan is taken for financing the aircraft",
        )
        self.options.declare(
            name="calculate_npax_design",
            default=False,
            types=bool,
            desc="True if NPAX_design is not provided",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]
        loan = self.options["loan"]

        if self.options["calculate_npax_design"]:
            self.add_subsystem(
                name="calculate_npax_design",
                subsys=LCCOperationalCalculateNPAXDesign(),
                promotes=["*"],
            )

        self.add_subsystem(
            name="passenger_per_flight",
            subsys=LCCOperationalPassengerPerFlight(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="annual_operational_revenue",
            subsys=LCCOperationalAnnualRevenue(years_of_service=years_of_service),
            promotes=["*"],
        )
        self.add_subsystem(
            name="revenue_per_rpk",
            subsys=LCCOperationalRevenuePerRPK(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="annual_energy_cost_projection",
            subsys=LCCOperationalAnnualEnergyCostProjection(years_of_service=years_of_service),
            promotes=["*"],
        )
        self.add_subsystem(
            name="annual_cash_flow",
            subsys=LCCOperationAnnualCashFlow(years_of_service=years_of_service, loan=loan),
            promotes=["*"],
        )
        self.add_subsystem(
            name="operational_npv_discount_factor",
            subsys=LCCNPVDiscountFactor(duration_in_years=years_of_service),
            promotes=[
                ("discount_rate", "data:cost:operation:discount_rate"),
            ],
        )
        self.add_subsystem(
            name="operational_net_present_value",
            subsys=LCCNetPresentValue(duration_in_years=years_of_service),
            promotes=[
                ("annual_net_cash_flow", "data:cost:operation:annual_cash_flow"),
                ("net_present_value", "data:cost:operation:net_present_value"),
            ],
        )
        self.add_subsystem(
            name="operational_profitability_index",
            subsys=LCCProfitabilityIndex(duration_in_years=years_of_service),
            promotes=[
                ("initial_investment", "data:cost:msp_per_unit"),
                ("net_present_value", "data:cost:operation:net_present_value"),
                ("profitability_index", "data:cost:operation:profitability_index"),
            ],
        )

        self.connect(
            "operational_npv_discount_factor.annual_discount_factor",
            "operational_net_present_value.annual_discount_factor",
        )

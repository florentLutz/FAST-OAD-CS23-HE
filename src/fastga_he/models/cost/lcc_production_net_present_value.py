# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om

from .lcc_learning_curve_discount import LCCLearningCurveDiscount
from .lcc_project_development_cost_distribution import LCCProjectDevelopmentCostDistribution
from .lcc_annual_sales_gross_profit import LCCAnnualSalesGrossProfit
from .lcc_annual_delivery_count import LCCAnnualDeliveryCount
from .lcc_production_annual_cash_flow import LCCProductionAnnualCashFlow
from .lcc_npv_discount_factor import LCCNPVDiscountFactor
from .lcc_net_present_value import LCCNetPresentValue


class LCCProductionNetPresentValue(om.Group):
    """
    Group collects all the Net Present Value (NPV) projection calculation.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.add_subsystem(
            name="learning_curve_discount",
            subsys=LCCLearningCurveDiscount(
                years_of_development=years_of_development, years_of_program=years_of_program
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="project_development_cost_distribution",
            subsys=LCCProjectDevelopmentCostDistribution(years_of_development=years_of_development),
            promotes=["*"],
        )

        self.add_subsystem(
            name="annual_sales_gross_profit",
            subsys=LCCAnnualSalesGrossProfit(
                years_of_development=years_of_development, years_of_program=years_of_program
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="annual_delivery_count",
            subsys=LCCAnnualDeliveryCount(
                years_of_development=years_of_development, years_of_program=years_of_program
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="production_annual_cash_flow",
            subsys=LCCProductionAnnualCashFlow(
                years_of_development=years_of_development, years_of_program=years_of_program
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="npv_discount_factor",
            subsys=LCCNPVDiscountFactor(duration_in_years=years_of_program),
            promotes=["*"],
        )

        self.add_subsystem(
            name="net_present_value",
            subsys=LCCNetPresentValue(duration_in_years=years_of_program),
            promotes=["*"],
        )

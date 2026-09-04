# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .lcc_production_cost import LCCProductionCost
from .lcc_operational_cost import LCCOperationalCost
from .lcc_production_profitability import LCCProductionProfitability
from .lcc_operational_profitability import LCCOperationalProfitability
from .lcc_profidibility_index_optim import LCCProfitabilityIndexOptimization


@oad.RegisterOpenMDAOSystem("fastga_he.lcc.legacy", domain=ModelDomain.OTHER)
class LCC(om.Group):
    """
    Group that collects all the LCC computations.
    """

    def initialize(self):
        self.options.declare(
            name="delivery_method",
            default="flight",
            desc="Method with which the aircraft will be brought from the assembly plant to the "
            "end user. Can be either flown or carried by train",
            allow_none=False,
            values=["flight", "train"],
        )
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the powertrain",
            allow_none=False,
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
        self.options.declare(
            "fix_revenue_per_rpk",
            types=bool,
            default=False,
            desc="If True, the revenue per RPK will be an input.",
        )
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
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years.",
        )

    def setup(self):
        self.add_subsystem(
            name="production_cost",
            subsys=LCCProductionCost(
                power_train_file_path=self.options["power_train_file_path"],
                delivery_method=self.options["delivery_method"],
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="operational_cost",
            subsys=LCCOperationalCost(
                power_train_file_path=self.options["power_train_file_path"],
                delivery_method=self.options["delivery_method"],
                loan=self.options["loan"],
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="production_profitability",
            subsys=LCCProductionProfitability(
                years_of_development=self.options["years_of_development"],
                years_of_program=self.options["years_of_program"],
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="operational_profitability",
            subsys=LCCOperationalProfitability(
                years_of_service=self.options["years_of_service"],
                loan=self.options["loan"],
                calculate_npax_design=self.options["calculate_npax_design"],
                fix_revenue_per_rpk=self.options["fix_revenue_per_rpk"],
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "profitability_index_optim",
            LCCProfitabilityIndexOptimization(
                duration_in_years=self.options["years_of_service"]
        ),
            promotes=[("profitability_index","data:cost:operation:profitability_index"),
                      ("profitability_index_factor",
                       "data:cost:operation:profitability_index_factor")],
        )

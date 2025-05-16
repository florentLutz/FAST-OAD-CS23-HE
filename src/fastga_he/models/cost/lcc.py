# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .lcc_production_cost import LCCProductionCost
from .lcc_operational_cost import LCCOperationalCost


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

    def setup(self):
        loan = self.options["loan"]
        delivery_method = self.options["delivery_method"]

        self.add_subsystem(
            name="production_cost",
            subsys=LCCProductionCost(
                power_train_file_path=self.options["power_train_file_path"],
                delivery_method=delivery_method,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="operational_cost",
            subsys=LCCOperationalCost(
                power_train_file_path=self.options["power_train_file_path"],
                loan=loan,
            ),
            promotes=["*"],
        )

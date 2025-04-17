# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .lcc_production_cost import LCCProductionCost
from .lcc_operation_cost import LCCOperationCost


@oad.RegisterOpenMDAOSystem("fastga_he.lcc.legacy", domain=ModelDomain.OTHER)
class LCC(om.Group):
    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )
        self.options.declare(
            name="complex_flap",
            default=False,
            types=bool,
            desc="True if complex flap system is selected in design",
        )
        self.options.declare(
            name="pressurized",
            default=False,
            types=bool,
            desc="True if the aircraft is pressurized",
        )
        self.options.declare(
            name="tapered_wing",
            default=False,
            types=bool,
            desc="True if the aircraft has tapered wing",
        )
        self.options.declare(
            name="loan",
            default=True,
            types=bool,
            desc="True if loan is taken for financing the aircraft",
        )

        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        power_train_file_path = self.options["power_train_file_path"]
        complex_flap = self.options["complex_flap"]
        pressurized = self.options["pressurized"]
        tapered_wing = self.options["tapered_wing"]
        loan = self.options["loan"]
        use_operational_mission = self.options["use_operational_mission"]

        self.add_subsystem(
            name="production_cost",
            subsys=LCCProductionCost(
                power_train_file_path=power_train_file_path,
                complex_flap=complex_flap,
                pressurized=pressurized,
                tapered_wing=tapered_wing,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="operation_cost",
            subsys=LCCOperationCost(
                power_train_file_path=power_train_file_path,
                loan=loan,
                use_operational_mission=use_operational_mission,
            ),
            promotes=["*"],
        )

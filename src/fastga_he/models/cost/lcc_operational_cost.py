# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
import fastga_he.models.propulsion.components as he_comp

from .lcc_landing_cost import LCCLandingCost
from .lcc_annual_insurance_cost import LCCAnnualInsuranceCost
from .lcc_daily_parking_cost import LCCDailyParkingCost
from .lcc_annual_airport_cost import LCCAnnualAirportCost
from .lcc_annual_loan_cost import LCCAnnualLoanCost
from .lcc_annual_depreciation import LCCAnnualDepreciation
from .lcc_maintenance_cost import LCCMaintenanceCost
from .lcc_maintenance_miscellaneous_cost import LCCMaintenanceMiscellaneousCost
from .lcc_annual_crew_cost import LCCAnnualCrewCost
from .lcc_operational_cost_sum import LCCSumOperationalCost


class LCCOperationalCost(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):
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
        self.configurator.load(self.options["power_train_file_path"])
        loan = self.options["loan"]

        self.add_subsystem(
            name="landing_cost_per_operation",
            subsys=LCCLandingCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="annual_crew_cost",
            subsys=LCCAnnualCrewCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="daily_parking_cost",
            subsys=LCCDailyParkingCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="annual_airport_cost",
            subsys=LCCAnnualAirportCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="annual_insurance_cost", subsys=LCCAnnualInsuranceCost(), promotes=["*"]
        )

        self.add_subsystem(
            name="annual_loan_cost", subsys=LCCAnnualLoanCost(loan=loan), promotes=["*"]
        )

        self.add_subsystem(
            name="annual_depreciation", subsys=LCCAnnualDepreciation(), promotes=["*"]
        )

        self.add_subsystem(
            name="airframe_maintenance",
            subsys=LCCMaintenanceCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="airframe_maintenance_miscellaneous",
            subsys=LCCMaintenanceMiscellaneousCost(),
            promotes=["*"],
        )

        # For the most part we can reuse what is done for the sizing, no need to write a new
        # function

        cost_components_type = []
        cost_components_name = []
        (
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            _,
            _,
        ) = self.configurator.get_sizing_element_lists()

        for (
            component_name,
            component_name_id,
            component_type,
            component_om_type,
        ) in zip(
            components_name,
            components_name_id,
            components_type,
            components_om_type,
        ):
            if hasattr(he_comp, "LCC" + component_om_type + "OperationalCost"):
                local_sub_sys = he_comp.__dict__["LCC" + component_om_type + "OperationalCost"]()
                local_sub_sys.options[component_name_id] = component_name
                cost_components_type.append(component_type)
                cost_components_name.append(component_name)

                self.add_subsystem(name=component_name, subsys=local_sub_sys, promotes=["*"])

        self.add_subsystem(
            name="cost_sum",
            subsys=LCCSumOperationalCost(
                cost_components_type=cost_components_type,
                cost_components_name=cost_components_name,
                loan=loan,
            ),
            promotes=["*"],
        )

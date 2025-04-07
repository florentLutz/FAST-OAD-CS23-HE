# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
import fastga_he.models.propulsion.components as he_comp

from .lcc_engineering_man_hours import LCCEngineeringManHours
from .lcc_tooling_man_hours import LCCToolingManHours
from .lcc_manufacturing_man_hours import LCCManufacturingManHours
from .lcc_engineering_cost import LCCEngineeringCost
from .lcc_tooling_cost import LCCToolingCost
from .lcc_manufacturing_cost import LCCManufacturingCost
from .lcc_dev_suppoet_cost import LCCDevSupportCost
from .lcc_quality_control_cost import LCCQualityControlCost
from .lcc_material_cost import LCCMaterialCost
from .lcc_flight_test_cost import LCCFlightTestCost
from .lcc_avionics_cost import LCCAvionicsCost
from .lcc_landing_gear_cost_reduction import LCCLandingGearCostReduction
from .lcc_certification_cost import LCCCertificationCost
from .lcc_msp import LCCMSP
from .lcc_production_cost_sum import LCCSumProductionCost


class LCCProductionCost(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

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

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])
        complex_flap = self.options["complex_flap"]
        pressurized = self.options["pressurized"]
        tapered_wing = self.options["tapered_wing"]

        # Calculate first the labor resources required for R&D and manufacturing of airframe
        self.add_subsystem(
            name="engineering_man_hours",
            subsys=LCCEngineeringManHours(complex_flap=complex_flap, pressurized=pressurized),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tooling_man_hours",
            subsys=LCCToolingManHours(
                complex_flap=complex_flap, pressurized=pressurized, tapered_wing=tapered_wing
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="manufacturing_man_hours",
            subsys=LCCManufacturingManHours(complex_flap=complex_flap),
            promotes=["*"],
        )

        # Calculate cost
        self.add_subsystem(
            name="engineering_cost_per_unit", subsys=LCCEngineeringCost(), promotes=["*"]
        )
        self.add_subsystem(name="tooling_cost_per_unit", subsys=LCCToolingCost(), promotes=["*"])
        self.add_subsystem(
            name="manufacturing_cost_per_unit", subsys=LCCManufacturingCost(), promotes=["*"]
        )
        self.add_subsystem(
            name="flight_test_cost_per_unit", subsys=LCCFlightTestCost(), promotes=["*"]
        )
        self.add_subsystem(
            name="quality_control_cost_per_unit", subsys=LCCQualityControlCost(), promotes=["*"]
        )

        self.add_subsystem(
            name="material_cost_per_unit",
            subsys=LCCMaterialCost(complex_flap=complex_flap, pressurized=pressurized),
            promotes=["*"],
        )

        self.add_subsystem(
            name="dev_support_cost_per_unit",
            subsys=LCCDevSupportCost(complex_flap=complex_flap, pressurized=pressurized),
            promotes=["*"],
        )

        self.add_subsystem(
            name="avionics_cost_per_unit",
            subsys=LCCAvionicsCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="certification_cost_per_unit",
            subsys=LCCCertificationCost(),
            promotes=["*"],
        )

        self.add_subsystem(
            name="landing_gear_cost_reduction",
            subsys=LCCLandingGearCostReduction(),
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
            if hasattr(he_comp, "LCC" + component_om_type + "Cost"):
                local_sub_sys = he_comp.__dict__["LCC" + component_om_type + "Cost"]()
                local_sub_sys.options[component_name_id] = component_name
                cost_components_type.append(component_type)
                cost_components_name.append(component_name)

                self.add_subsystem(name=component_name, subsys=local_sub_sys, promotes=["*"])

        self.add_subsystem(
            name="cost_sum",
            subsys=LCCSumProductionCost(
                cost_components_type=cost_components_type, cost_components_name=cost_components_name
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="sale_price",
            subsys=LCCMSP(),
            promotes=["*"],
        )

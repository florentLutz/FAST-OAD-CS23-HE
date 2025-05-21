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
from .lcc_dev_support_cost import LCCDevSupportCost
from .lcc_quality_control_cost import LCCQualityControlCost
from .lcc_material_cost import LCCMaterialCost
from .lcc_flight_test_cost import LCCFlightTestCost
from .lcc_avionics_cost import LCCAvionicsCost
from .lcc_landing_gear_cost_reduction import LCCLandingGearCostReduction
from .lcc_certification_cost import LCCCertificationCost
from .lcc_msp import LCCMSP
from .lcc_production_cost_sum import LCCSumProductionCost
from .lcc_fuel_cost import LCCFuelCost
from .lcc_electricity_cost import LCCElectricityCost
from .lcc_delivery_cost import LCCDeliveryCost
from .lcc_deliveray_duration_ratio import LCCDeliveryDurationRatio
from .lcc_leaarning_curve_factor import LCCLearningCurveFactor
from .lcc_leaarning_curve_discount import LCCLearningCurveDiscount

from .constants import ELECTRICITY_STORAGE_TYPES


class LCCProductionCost(om.Group):
    """
    Group collects all the production cost calculation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

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
            name="learning_curve",
            default=False,
            desc="Learning curve for providing the discount rate of the manufacturing and tooling "
            "man hours.",
        )
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])
        delivery_method = self.options["delivery_method"]

        # Calculate first the labor resources required for R&D and manufacturing of airframe
        self.add_subsystem(
            name="engineering_man_hours_5_years",
            subsys=LCCEngineeringManHours(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tooling_man_hours_5_years",
            subsys=LCCToolingManHours(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="manufacturing_man_hours_5_years",
            subsys=LCCManufacturingManHours(),
            promotes=["*"],
        )

        if self.options["learning_curve"]:
            self.add_subsystem(
                name="learning_curve_factor",
                subsys=LCCLearningCurveFactor(),
                promotes=["*"],
            )
            self.add_subsystem(
                name="learning_curve_discount",
                subsys=LCCLearningCurveDiscount(),
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
            subsys=LCCMaterialCost(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="dev_support_cost_per_unit",
            subsys=LCCDevSupportCost(),
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
        electricity_components_type = []
        electricity_components_name = []
        (
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            _,
            _,
        ) = self.configurator.get_sizing_element_lists()

        tank_names, tank_types, fuel_types = self.configurator.get_fuel_tank_list_and_fuel()

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

            if components_type in ELECTRICITY_STORAGE_TYPES:
                electricity_components_type.append(component_type)
                electricity_components_name.append(component_name)

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

        if delivery_method == "flight":
            self.add_subsystem(
                name="fuel_cost",
                subsys=LCCFuelCost(
                    tank_types=tank_types, tank_names=tank_names, fuel_types=fuel_types
                ),
                promotes=["*"],
            )
            self.add_subsystem(
                name="electric_energy_cost",
                subsys=LCCElectricityCost(
                    electricity_components_type=electricity_components_type,
                    electricity_components_name=electricity_components_name,
                ),
                promotes=["*"],
            )
            self.add_subsystem(
                name="delivery_duration_ratio",
                subsys=LCCDeliveryDurationRatio(),
                promotes=["*"],
            )

        self.add_subsystem(
            name="delivery_cost",
            subsys=LCCDeliveryCost(delivery_method=delivery_method),
            promotes=["*"],
        )

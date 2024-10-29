# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import openmdao.api as om

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
import fastga_he.models.propulsion.components as he_comp

from .lca_aircraft_per_fu import LCAAircraftPerFU
from .lca_use_flight_per_fu import LCAUseFlightPerFU

from .lca_line_test_mission_ratio import LCARatioTestFlightMission

from .lca_wing_weight_per_fu import LCAWingWeightPerFU
from .lca_fuselage_weight_per_fu import LCAFuselageWeightPerFU
from .lca_htp_weight_per_fu import LCAHTPWeightPerFU
from .lca_vtp_weight_per_fu import LCAVTPWeightPerFU
from .lca_landing_gear_weight_per_fu import LCALandingGearWeightPerFU
from .lca_flight_control_weight_per_fu import LCAFlightControlsWeightPerFU
from .lca_empty_aircraft_weight_per_fu import LCAEmptyAircraftWeightPerFU

from .lca_gasoline_per_fu import LCAGasolinePerFU
from .lca_kerosene_per_fu import LCAKerosenePerFU
from .lca_electricty_per_fu import LCAElectricityPerFU

from .lca_core import LCACore, METHODS_TO_FILE


class LCA(om.Group):
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
            name="component_level_breakdown",
            default=False,
            types=bool,
            desc="If true in addition to a breakdown, phase by phase, adds a breakdown component "
            "by component",
        )
        self.options.declare(
            name="impact_assessment_method",
            default="ReCiPe 2016 v1.03",
            desc="Impact assessment method to be used",
            values=list(METHODS_TO_FILE.keys()),
        )
        self.options.declare(
            name="ecoinvent_version",
            default="3.9.1",
            desc="EcoInvent version to use",
            values=["3.9.1"],
        )
        self.options.declare(
            name="airframe_material",
            default="aluminium",
            desc="Material used for the airframe which include wing, fuselage, HTP and VTP. LG will"
            " always be in aluminium and flight controls in steel",
            allow_none=False,
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])

        self.add_subsystem(name="aircraft_per_fu", subsys=LCAAircraftPerFU(), promotes=["*"])
        self.add_subsystem(name="flight_per_fu", subsys=LCAUseFlightPerFU(), promotes=["*"])

        self.add_subsystem(
            name="line_tests_mission_ratio", subsys=LCARatioTestFlightMission(), promotes=["*"]
        )

        # Adds all the LCA groups for the airframe which will be here regardless of the powertrain
        self.add_subsystem(name="pre_lca_wing", subsys=LCAWingWeightPerFU(), promotes=["*"])
        self.add_subsystem(name="pre_lca_fuselage", subsys=LCAFuselageWeightPerFU(), promotes=["*"])
        self.add_subsystem(name="pre_lca_htp", subsys=LCAHTPWeightPerFU(), promotes=["*"])
        self.add_subsystem(name="pre_lca_vtp", subsys=LCAVTPWeightPerFU(), promotes=["*"])
        self.add_subsystem(name="pre_lca_lg", subsys=LCALandingGearWeightPerFU(), promotes=["*"])
        self.add_subsystem(
            name="pre_lca_flight_control", subsys=LCAFlightControlsWeightPerFU(), promotes=["*"]
        )
        self.add_subsystem(
            name="pre_lca_empty_aircraft", subsys=LCAEmptyAircraftWeightPerFU(), promotes=["*"]
        )

        # For the most part we can reuse what is done for the sizing, no need to write a new
        # function
        (
            components_name,
            components_name_id,
            components_type,
            components_om_type,
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
            local_sub_sys = he_comp.__dict__["PreLCA" + component_om_type]()
            local_sub_sys.options[component_name_id] = component_name

            self.add_subsystem(name=component_name, subsys=local_sub_sys, promotes=["*"])

        # Add the component to compute kerosene/gasoline/electricity_per_fu if needed

        gasoline_tank_names, gasoline_tank_types = [], []
        kerosene_tank_names, kerosene_tank_types = [], []

        tank_names, tank_types, contents = self.configurator.get_fuel_tank_list_and_fuel()

        for tank_name, tank_type, content in zip(tank_names, tank_types, contents):
            if content == "jet_fuel":
                kerosene_tank_names.append(tank_name)
                kerosene_tank_types.append(tank_type)
            elif content == "avgas":
                gasoline_tank_names.append(tank_name)
                gasoline_tank_types.append(tank_type)

        if gasoline_tank_names:
            self.add_subsystem(
                name="pre_lca_avgas",
                subsys=LCAGasolinePerFU(
                    tanks_name_list=gasoline_tank_names, tanks_type_list=gasoline_tank_types
                ),
                promotes=["*"],
            )

        if kerosene_tank_types:
            self.add_subsystem(
                name="pre_lca_kerosene",
                subsys=LCAKerosenePerFU(
                    tanks_name_list=kerosene_tank_names, tanks_type_list=kerosene_tank_types
                ),
                promotes=["*"],
            )

        battery_names, battery_types = self.configurator.get_battery_list()

        if battery_names:
            self.add_subsystem(
                name="pre_lca_electricity",
                subsys=LCAElectricityPerFU(
                    batteries_name_list=battery_names, batteries_type_list=battery_types
                ),
                promotes=["*"],
            )

        self.add_subsystem(
            name="lca_core",
            subsys=LCACore(
                power_train_file_path=self.options["power_train_file_path"],
                component_level_breakdown=self.options["component_level_breakdown"],
                impact_assessment_method=self.options["impact_assessment_method"],
                ecoinvent_version=self.options["ecoinvent_version"],
                airframe_material=self.options["airframe_material"],
            ),
            promotes=["*"],
        )

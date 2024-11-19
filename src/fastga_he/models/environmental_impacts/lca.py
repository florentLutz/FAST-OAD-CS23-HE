# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastga_he.models.propulsion.components as he_comp
from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
from .lca_aircraft_per_fu import LCAAircraftPerFU
from .lca_core import LCACore
from .lca_core_normalization import LCACoreNormalisation
from .lca_core_weighting import LCACoreWeighting
from .lca_core_aggregation import LCACoreAggregation
from .lca_delivery_mission_ratio import LCARatioDeliveryFlightMission
from .lca_distribution_cargo import LCADistributionCargoMassDistancePerFU
from .lca_electricty_per_fu import LCAElectricityPerFU
from .lca_empty_aircraft_weight_per_fu import LCAEmptyAircraftWeightPerFU
from .lca_flight_control_weight_per_fu import LCAFlightControlsWeightPerFU
from .lca_fuselage_weight_per_fu import LCAFuselageWeightPerFU
from .lca_gasoline_per_fu import LCAGasolinePerFU
from .lca_htp_weight_per_fu import LCAHTPWeightPerFU
from .lca_kerosene_per_fu import LCAKerosenePerFU
from .lca_landing_gear_weight_per_fu import LCALandingGearWeightPerFU
from .lca_line_test_mission_ratio import LCARatioTestFlightMission
from .lca_use_flight_per_fu import LCAUseFlightPerFU
from .lca_vtp_weight_per_fu import LCAVTPWeightPerFU
from .lca_wing_weight_per_fu import LCAWingWeightPerFU
from .resources.constants import (
    METHODS_TO_FILE,
    METHODS_TO_NORMALIZATION,
    METHODS_TO_WEIGHTING,
    LCA_PREFIX,
    NORMALIZATION_FACTOR,
    WEIGHTING_FACTOR,
)


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
        self.options.declare(
            name="delivery_method",
            default="flight",
            desc="Method with which the aircraft will be brought from the assembly plant to the "
            "end user. Can be either flown or carried by train",
            allow_none=False,
            values=["flight", "train"],
        )
        self.options.declare(
            name="electric_mix",
            default="default",
            desc="By default to construct the aircraft, a European electric mix is used. This "
            "forces all higher level process to use a different mix. This will not affect "
            "subprocesses of proxies directly taken from EcoInvent",
            allow_none=False,
            values=["default", "french", "slovenia"],
        )
        self.options.declare(
            name="normalization",
            default=False,
            types=bool,
            desc="If available, the normalization step will be added to the LCA process",
        )
        self.options.declare(
            name="weighting",
            default=False,
            types=bool,
            desc="If available, the weighting and aggregation steps will be added to the LCA process",
        )
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])

        self.add_subsystem(
            name="aircraft_per_fu",
            subsys=LCAAircraftPerFU(
                use_operational_mission=self.options["use_operational_mission"]
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="flight_per_fu",
            subsys=LCAUseFlightPerFU(
                use_operational_mission=self.options["use_operational_mission"]
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="line_tests_mission_ratio", subsys=LCARatioTestFlightMission(), promotes=["*"]
        )
        # Will always be added even when not used because I can't see a smarter way to compute
        # fuel and emissions that doing as I did for line tests
        self.add_subsystem(
            name="delivery_mission_ratio", subsys=LCARatioDeliveryFlightMission(), promotes=["*"]
        )
        if self.options["delivery_method"] == "train":
            # Will only be used if this method is added which contrast with the other method
            self.add_subsystem(
                name="delivery_ton_km",
                subsys=LCADistributionCargoMassDistancePerFU(),
                promotes=["*"],
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
                delivery_method=self.options["delivery_method"],
                electric_mix=self.options["electric_mix"],
            ),
            promotes=["*"],
        )

        if (
            self.options["normalization"]
            and METHODS_TO_NORMALIZATION[self.options["impact_assessment_method"]]
        ):
            self.add_subsystem(
                name="lca_normalization",
                subsys=LCACoreNormalisation(),
                promotes=["*"],
            )

            ivc = om.IndepVarComp()
            self.add_subsystem(name="lca_normalization_factor", subsys=ivc, promotes=["*"])

        # Be careful here, the weighting step should only be done if the normalization step has
        # been done beforehand

        if (
            self.options["normalization"]
            and self.options["weighting"]
            and METHODS_TO_WEIGHTING[self.options["impact_assessment_method"]]
        ):
            self.add_subsystem(
                name="lca_weighting",
                subsys=LCACoreWeighting(),
                promotes=["*"],
            )
            self.add_subsystem(
                name="lca_aggregation",
                subsys=LCACoreAggregation(),
                promotes=["*"],
            )

    def configure(self):
        if (
            self.options["normalization"]
            and METHODS_TO_NORMALIZATION[self.options["impact_assessment_method"]]
        ):
            normalization_inputs_list = self.lca_core.list_outputs(
                return_format="dict", out_stream=None
            ).keys()
            self.lca_normalization.inputs_list = normalization_inputs_list

            added_normalization_factor = []
            self.lca_normalization.normalization_factor = NORMALIZATION_FACTOR[
                self.options["impact_assessment_method"]
            ]

            for var_in in normalization_inputs_list:
                # We transform the name of the input variable in the following manner: we replace
                # the impact name with (impact_name)_normalized

                method_name = var_in.split(":")[2]

                # Normalize only if the normalization factor exists, which might not be the case
                # for recipe and total_ impacts
                if method_name in NORMALIZATION_FACTOR[self.options["impact_assessment_method"]]:
                    normalized_method_name = method_name + "_normalized"
                    normalization_factor_name = LCA_PREFIX + method_name + ":normalization_factor"
                    normalization_factor = NORMALIZATION_FACTOR[
                        self.options["impact_assessment_method"]
                    ][method_name]
                    var_out = var_in.replace(method_name, normalized_method_name)

                    # Add outputs from core LCIA as inputs to normalization
                    self.lca_normalization.add_input(var_in, val=np.nan, units=None)
                    self.lca_normalization.add_output(var_out, units=None)
                    self.lca_normalization.declare_partials(
                        of=var_out, wrt=var_in, val=1.0 / normalization_factor
                    )

                    # Declaring normalization factor in a separate ivc so that it doesn't become an
                    # input of lca_normalization and the partials are easier to declare
                    if normalization_factor_name not in added_normalization_factor:
                        self.lca_normalization_factor.add_output(
                            normalization_factor_name, val=normalization_factor, units=None
                        )
                        added_normalization_factor.append(normalization_factor_name)

        if (
            self.options["normalization"]
            and self.options["weighting"]
            and METHODS_TO_WEIGHTING[self.options["impact_assessment_method"]]
        ):
            potential_weighting_inputs_list = self.lca_normalization.list_outputs(
                return_format="dict", out_stream=None
            ).keys()
            self.lca_weighting.inputs_list = potential_weighting_inputs_list
            self.lca_weighting.weighting_factor = WEIGHTING_FACTOR[
                self.options["impact_assessment_method"]
            ]

            added_weighting_factor = []

            self.lca_aggregation.add_output(
                "data:environmental_impact:single_score", val=0.0, units=None
            )

            for var_in in potential_weighting_inputs_list:
                # We transform the name of the input variable in the following manner: we replace
                # the impact name with (impact_name)_weighted

                normalized_method_name = var_in.split(":")[2]
                method_name = normalized_method_name.replace("_normalized", "")

                if method_name in WEIGHTING_FACTOR[self.options["impact_assessment_method"]]:
                    weighted_method_name = method_name + "_weighted"
                    weighting_factor_name = LCA_PREFIX + method_name + ":weighting_factor"
                    weighting_factor = WEIGHTING_FACTOR[self.options["impact_assessment_method"]][
                        method_name
                    ]
                    var_out = var_in.replace(normalized_method_name, weighted_method_name)

                    self.lca_weighting.add_input(var_in, val=np.nan, units=None)

                    if weighting_factor_name not in added_weighting_factor:
                        self.lca_weighting.add_input(
                            weighting_factor_name, val=weighting_factor, units=None
                        )
                        added_weighting_factor.append(weighting_factor_name)

                    self.lca_weighting.add_output(var_out, units=None)
                    self.lca_weighting.declare_partials(
                        of=var_out, wrt=[var_in, weighting_factor_name], method="exact"
                    )

                    # For the single score we only take the sum of each category meaning the
                    # variable that match the following pattern :
                    if var_out == LCA_PREFIX + weighted_method_name + ":sum":
                        self.lca_aggregation.add_input(var_out, units=None)
                        self.lca_aggregation.declare_partials(
                            of="data:environmental_impact:single_score", wrt=var_out, val=1.0
                        )

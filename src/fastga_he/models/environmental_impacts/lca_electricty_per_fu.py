# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .constants import SERVICE_ELECTRICITY_PER_FU

SUBMODEL_ELECTRICITY_PER_FU_FROM_ONE_MISSION = (
    "fastga_he.submodel.lca.energy_per_fu.from_one_mission"
)
SUBMODEL_ELECTRICITY_PER_FU_FROM_ONE_USE_PHASE = (
    "fastga_he.submodel.lca.energy_per_fu.from_use_phase_value"
)


oad.RegisterSubmodel.active_models[SERVICE_ELECTRICITY_PER_FU] = (
    SUBMODEL_ELECTRICITY_PER_FU_FROM_ONE_MISSION
)


@oad.RegisterSubmodel(SERVICE_ELECTRICITY_PER_FU, SUBMODEL_ELECTRICITY_PER_FU_FROM_ONE_MISSION)
class LCAElectricityPerFU(om.ExplicitComponent):
    """
    Computes the electricity required per functional unit for the use, distribution and line testing
    phases. Computed based on the energy required for one flight and the number of FU in one flight.
    """

    def initialize(self):
        self.options.declare(
            name="batteries_name_list",
            default=None,
            types=list,
            desc="List of names of the batteries, inside the powertrain, that store electricity. "
            "The term battery actually refer here to any component that stores electricity",
            allow_none=False,
        )
        self.options.declare(
            name="batteries_type_list",
            default=None,
            types=list,
            desc="List of types of the batteries, inside the powertrain, that store electricity",
            allow_none=False,
        )

    def setup(self):
        batteries_names = self.options["batteries_name_list"]
        batteries_types = self.options["batteries_type_list"]

        self.add_input(name="data:environmental_impact:flight_per_fu", val=1e-3)
        self.add_input(name="data:environmental_impact:aircraft_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:line_test:mission_ratio", val=np.nan)
        self.add_input(name="data:environmental_impact:delivery:mission_ratio", val=np.nan)

        self.add_output(
            name="data:LCA:operation:he_power_train:electricity:energy_per_fu", units="W*h", val=0.0
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:electricity:energy_per_fu",
            wrt="data:environmental_impact:flight_per_fu",
            method="exact",
        )

        self.add_output(
            name="data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            units="W*h",
            val=0.0,
        )
        self.declare_partials(
            of="data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:line_test:mission_ratio",
            ],
            method="exact",
        )

        self.add_output(
            name="data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            units="W*h",
            val=0.0,
        )
        self.declare_partials(
            of="data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:delivery:mission_ratio",
            ],
            method="exact",
        )

        for batteries_name, batteries_type in zip(batteries_names, batteries_types):
            input_name = (
                "data:propulsion:he_power_train:"
                + batteries_type
                + ":"
                + batteries_name
                + ":energy_consumed_main_route"
            )
            self.add_input(input_name, units="W*h", val=np.nan)

            self.declare_partials(of="*", wrt=input_name, method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        batteries_names = self.options["batteries_name_list"]
        batteries_types = self.options["batteries_type_list"]

        total_fuel = 0

        for batteries_name, batteries_type in zip(batteries_names, batteries_types):
            total_fuel += inputs[
                "data:propulsion:he_power_train:"
                + batteries_type
                + ":"
                + batteries_name
                + ":energy_consumed_main_route"
            ]

        outputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"] = (
            total_fuel * inputs["data:environmental_impact:flight_per_fu"]
        )
        outputs["data:LCA:manufacturing:he_power_train:electricity:energy_per_fu"] = (
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * total_fuel
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )
        outputs["data:LCA:distribution:he_power_train:electricity:energy_per_fu"] = (
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * total_fuel
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        batteries_names = self.options["batteries_name_list"]
        batteries_types = self.options["batteries_type_list"]

        partial_flight_per_fu = 0

        for batteries_name, batteries_type in zip(batteries_names, batteries_types):
            partials[
                "data:LCA:operation:he_power_train:electricity:energy_per_fu",
                "data:propulsion:he_power_train:"
                + batteries_type
                + ":"
                + batteries_name
                + ":energy_consumed_main_route",
            ] = inputs["data:environmental_impact:flight_per_fu"]

            partials[
                "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
                "data:propulsion:he_power_train:"
                + batteries_type
                + ":"
                + batteries_name
                + ":energy_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

            partials[
                "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
                "data:propulsion:he_power_train:"
                + batteries_type
                + ":"
                + batteries_name
                + ":energy_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

            partial_flight_per_fu += inputs[
                "data:propulsion:he_power_train:"
                + batteries_type
                + ":"
                + batteries_name
                + ":energy_consumed_main_route"
            ]

        partials[
            "data:LCA:operation:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:flight_per_fu",
        ] = partial_flight_per_fu

        partials[
            "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:environmental_impact:line_test:mission_ratio"] * partial_flight_per_fu
        partials[
            "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:line_test:mission_ratio",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * partial_flight_per_fu

        partials[
            "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:environmental_impact:delivery:mission_ratio"] * partial_flight_per_fu
        partials[
            "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:delivery:mission_ratio",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * partial_flight_per_fu


@oad.RegisterSubmodel(SERVICE_ELECTRICITY_PER_FU, SUBMODEL_ELECTRICITY_PER_FU_FROM_ONE_USE_PHASE)
class LCAElectricityPerFUFromUsePhaseValue(om.ExplicitComponent):
    """
    Computes the electricity required per functional unit for the distribution and line testing
    phases based on the value for the use phase which is considered an input.
    """

    def initialize(self):
        # For compatibility reason but actually not used.
        self.options.declare(
            name="batteries_name_list",
            default=None,
            types=list,
            desc="List of names of the batteries, inside the powertrain, that store electricity. "
            "The term battery actually refer here to any component that stores electricity",
            allow_none=False,
        )
        self.options.declare(
            name="batteries_type_list",
            default=None,
            types=list,
            desc="List of types of the batteries, inside the powertrain, that store electricity",
            allow_none=False,
        )

    def setup(self):
        self.add_input(name="data:environmental_impact:flight_per_fu", val=1e-3)
        self.add_input(name="data:environmental_impact:aircraft_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:line_test:mission_ratio", val=np.nan)
        self.add_input(name="data:environmental_impact:delivery:mission_ratio", val=np.nan)
        self.add_input(
            name="data:LCA:operation:he_power_train:electricity:energy_per_fu",
            units="W*h",
            val=np.nan,
        )

        self.add_output(
            name="data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            units="W*h",
            val=0.0,
        )
        self.declare_partials(
            of="data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            wrt=[
                "data:LCA:operation:he_power_train:electricity:energy_per_fu",
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:flight_per_fu",
                "data:environmental_impact:line_test:mission_ratio",
            ],
            method="exact",
        )

        self.add_output(
            name="data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            units="W*h",
            val=0.0,
        )
        self.declare_partials(
            of="data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            wrt=[
                "data:LCA:operation:he_power_train:electricity:energy_per_fu",
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:flight_per_fu",
                "data:environmental_impact:delivery:mission_ratio",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        electricity_per_fu_operation = inputs[
            "data:LCA:operation:he_power_train:electricity:energy_per_fu"
        ]

        outputs["data:LCA:manufacturing:he_power_train:electricity:energy_per_fu"] = (
            electricity_per_fu_operation
            / inputs["data:environmental_impact:flight_per_fu"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:environmental_impact:line_test:mission_ratio"]
        )
        outputs["data:LCA:distribution:he_power_train:electricity:energy_per_fu"] = (
            electricity_per_fu_operation
            / inputs["data:environmental_impact:flight_per_fu"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:environmental_impact:delivery:mission_ratio"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            "data:LCA:operation:he_power_train:electricity:energy_per_fu",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:environmental_impact:line_test:mission_ratio"]
            / inputs["data:environmental_impact:flight_per_fu"]
        )
        partials[
            "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * inputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"]
            / inputs["data:environmental_impact:flight_per_fu"]
        )
        partials[
            "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:line_test:mission_ratio",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"]
            / inputs["data:environmental_impact:flight_per_fu"]
        )
        partials[
            "data:LCA:manufacturing:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:flight_per_fu",
        ] = -(
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"]
            / inputs["data:environmental_impact:flight_per_fu"] ** 2.0
        )

        partials[
            "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            "data:LCA:operation:he_power_train:electricity:energy_per_fu",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:environmental_impact:delivery:mission_ratio"]
            / inputs["data:environmental_impact:flight_per_fu"]
        )
        partials[
            "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * inputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"]
            / inputs["data:environmental_impact:flight_per_fu"]
        )
        partials[
            "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:delivery:mission_ratio",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"]
            / inputs["data:environmental_impact:flight_per_fu"]
        )
        partials[
            "data:LCA:distribution:he_power_train:electricity:energy_per_fu",
            "data:environmental_impact:flight_per_fu",
        ] = -(
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * inputs["data:LCA:operation:he_power_train:electricity:energy_per_fu"]
            / inputs["data:environmental_impact:flight_per_fu"] ** 2.0
        )

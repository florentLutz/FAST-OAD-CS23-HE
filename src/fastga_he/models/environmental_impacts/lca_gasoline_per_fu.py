# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAGasolinePerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="tanks_name_list",
            default=None,
            types=list,
            desc="List of names of the tanks, inside the powertrain, that store gasoline",
            allow_none=False,
        )
        self.options.declare(
            name="tanks_type_list",
            default=None,
            types=list,
            desc="List of types of the tanks, inside the powertrain, that store gasoline",
            allow_none=False,
        )

    def setup(self):
        tanks_names = self.options["tanks_name_list"]
        tanks_types = self.options["tanks_type_list"]

        self.add_input(name="data:environmental_impact:flight_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:aircraft_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:line_test:mission_ratio", val=np.nan)
        self.add_input(name="data:environmental_impact:delivery:mission_ratio", val=np.nan)

        self.add_output(
            name="data:LCA:operation:he_power_train:gasoline:mass_per_fu", units="kg", val=0.0
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:gasoline:mass_per_fu",
            wrt="data:environmental_impact:flight_per_fu",
            method="exact",
        )

        self.add_output(
            name="data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu", units="kg", val=0.0
        )
        self.declare_partials(
            of="data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:line_test:mission_ratio",
            ],
            method="exact",
        )

        self.add_output(
            name="data:LCA:distribution:he_power_train:gasoline:mass_per_fu",
            units="kg",
            val=0.0,
        )
        self.declare_partials(
            of="data:LCA:distribution:he_power_train:gasoline:mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:delivery:mission_ratio",
            ],
            method="exact",
        )

        for tank_name, tank_type in zip(tanks_names, tanks_types):
            input_name = (
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_mission"
            )
            self.add_input(input_name, units="kg", val=np.nan)

            self.declare_partials(of="*", wrt=input_name, method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tanks_names = self.options["tanks_name_list"]
        tanks_types = self.options["tanks_type_list"]

        total_fuel = 0

        for tank_name, tank_type in zip(tanks_names, tanks_types):
            total_fuel += inputs[
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_mission"
            ]

        outputs["data:LCA:operation:he_power_train:gasoline:mass_per_fu"] = (
            total_fuel * inputs["data:environmental_impact:flight_per_fu"]
        )

        outputs["data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu"] = (
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * total_fuel
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )
        outputs["data:LCA:distribution:he_power_train:gasoline:mass_per_fu"] = (
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * total_fuel
            * inputs["data:environmental_impact:aircraft_per_fu"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tanks_names = self.options["tanks_name_list"]
        tanks_types = self.options["tanks_type_list"]

        partial_flight_per_fu = 0

        for tank_name, tank_type in zip(tanks_names, tanks_types):
            partials[
                "data:LCA:operation:he_power_train:gasoline:mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_mission",
            ] = inputs["data:environmental_impact:flight_per_fu"]

            partials[
                "data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_mission",
            ] = (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

            partials[
                "data:LCA:distribution:he_power_train:gasoline:mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_mission",
            ] = (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )

            partial_flight_per_fu += inputs[
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_mission"
            ]

        partials[
            "data:LCA:operation:he_power_train:gasoline:mass_per_fu",
            "data:environmental_impact:flight_per_fu",
        ] = partial_flight_per_fu

        partials[
            "data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:environmental_impact:line_test:mission_ratio"] * partial_flight_per_fu
        partials[
            "data:LCA:manufacturing:he_power_train:gasoline:mass_per_fu",
            "data:environmental_impact:line_test:mission_ratio",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * partial_flight_per_fu

        partials[
            "data:LCA:distribution:he_power_train:gasoline:mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:environmental_impact:delivery:mission_ratio"] * partial_flight_per_fu
        partials[
            "data:LCA:distribution:he_power_train:gasoline:mass_per_fu",
            "data:environmental_impact:delivery:mission_ratio",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * partial_flight_per_fu

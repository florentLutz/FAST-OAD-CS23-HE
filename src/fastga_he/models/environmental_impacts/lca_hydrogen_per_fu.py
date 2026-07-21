#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAHydrogenPerFU(om.ExplicitComponent):
    """
    Computation of the mass of hydrogen necessary to fulfil the functional unit, considering all
    stage of the life cycle. Hydrogen leakage throughout the entire infrastructure will also be
    considered using a ratio of the total throughput. Default value taken from :cite:`evon:2026`.

    Could be tuned to zero or close if local production is assumed.
    """

    def initialize(self):
        self.options.declare(
            name="tanks_name_list",
            default=None,
            types=list,
            desc="List of names of the tanks, inside the powertrain, that store hydrogen",
            allow_none=False,
        )
        self.options.declare(
            name="tanks_type_list",
            default=None,
            types=list,
            desc="List of types of the tanks, inside the powertrain, that store hydrogen",
            allow_none=False,
        )

    def setup(self):
        tanks_names = self.options["tanks_name_list"]
        tanks_types = self.options["tanks_type_list"]

        self.add_input(name="data:environmental_impact:flight_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:aircraft_per_fu", val=np.nan)
        self.add_input(name="data:environmental_impact:line_test:mission_ratio", val=np.nan)
        self.add_input(name="data:environmental_impact:delivery:mission_ratio", val=np.nan)

        self.add_input(
            name="data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            val=3.0,
            units="percent",
            desc="Percentage of total hydrogen throughput leaking throughout the entire "
            "infrastructure (production, distribution, ...)",
        )

        self.add_output(
            name="data:LCA:operation:he_power_train:hydrogen:mass_per_fu", units="kg", val=0.0
        )
        self.add_output(
            name="data:LCA:operation:he_power_train:hydrogen:leakage_mass_per_fu",
            units="kg",
            val=0.0,
        )

        self.add_output(
            name="data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu", units="kg", val=0.0
        )
        self.add_output(
            name="data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu",
            units="kg",
            val=0.0,
        )

        self.add_output(
            name="data:LCA:distribution:he_power_train:hydrogen:mass_per_fu",
            units="kg",
            val=0.0,
        )
        self.add_output(
            name="data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu",
            units="kg",
            val=0.0,
        )

        for tank_name, tank_type in zip(tanks_names, tanks_types):
            input_name = (
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route"
            )
            self.add_input(input_name, units="kg", val=np.nan)

    def setup_partials(self):
        tanks_names = self.options["tanks_name_list"]
        tanks_types = self.options["tanks_type_list"]

        self.declare_partials(
            of="data:LCA:operation:he_power_train:hydrogen:mass_per_fu",
            wrt=[
                "data:environmental_impact:flight_per_fu",
                "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:hydrogen:leakage_mass_per_fu",
            wrt=[
                "data:environmental_impact:flight_per_fu",
                "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            ],
            method="exact",
        )

        self.declare_partials(
            of="data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:line_test:mission_ratio",
                "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:line_test:mission_ratio",
                "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            ],
            method="exact",
        )

        self.declare_partials(
            of="data:LCA:distribution:he_power_train:hydrogen:mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:delivery:mission_ratio",
                "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:environmental_impact:delivery:mission_ratio",
                "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
            ],
            method="exact",
        )

        for tank_name, tank_type in zip(tanks_names, tanks_types):
            input_name = (
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route"
            )
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
                + ":fuel_consumed_main_route"
            ]

        leak_percentage = (
            inputs["data:environmental_impact:hydrogen:infrastructure_leak_percentage"] / 100.0
        )

        outputs["data:LCA:operation:he_power_train:hydrogen:mass_per_fu"] = (
            total_fuel * inputs["data:environmental_impact:flight_per_fu"]
        ) / (1.0 - leak_percentage)
        outputs["data:LCA:operation:he_power_train:hydrogen:leakage_mass_per_fu"] = (
            (total_fuel * inputs["data:environmental_impact:flight_per_fu"])
            * leak_percentage
            / (1.0 - leak_percentage)
        )

        outputs["data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu"] = (
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * total_fuel
            * inputs["data:environmental_impact:aircraft_per_fu"]
        ) / (1.0 - leak_percentage)
        outputs["data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu"] = (
            (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * total_fuel
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )
            * leak_percentage
            / (1.0 - leak_percentage)
        )

        outputs["data:LCA:distribution:he_power_train:hydrogen:mass_per_fu"] = (
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * total_fuel
            * inputs["data:environmental_impact:aircraft_per_fu"]
        ) / (1.0 - leak_percentage)
        outputs["data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu"] = (
            (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * total_fuel
                * inputs["data:environmental_impact:aircraft_per_fu"]
            )
            * leak_percentage
            / (1.0 - leak_percentage)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tanks_names = self.options["tanks_name_list"]
        tanks_types = self.options["tanks_type_list"]

        partial_flight_per_fu = 0

        leak_percentage = (
            inputs["data:environmental_impact:hydrogen:infrastructure_leak_percentage"] / 100.0
        )

        for tank_name, tank_type in zip(tanks_names, tanks_types):
            partials[
                "data:LCA:operation:he_power_train:hydrogen:mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route",
            ] = inputs["data:environmental_impact:flight_per_fu"] / (1.0 - leak_percentage)
            partials[
                "data:LCA:operation:he_power_train:hydrogen:leakage_mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:flight_per_fu"]
                * leak_percentage
                / (1.0 - leak_percentage)
            )

            partials[
                "data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
                / (1.0 - leak_percentage)
            )
            partials[
                "data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:line_test:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
                * leak_percentage
                / (1.0 - leak_percentage)
            )

            partials[
                "data:LCA:distribution:he_power_train:hydrogen:mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
                / (1.0 - leak_percentage)
            )
            partials[
                "data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu",
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route",
            ] = (
                inputs["data:environmental_impact:delivery:mission_ratio"]
                * inputs["data:environmental_impact:aircraft_per_fu"]
                * leak_percentage
                / (1.0 - leak_percentage)
            )

            partial_flight_per_fu += inputs[
                "data:propulsion:he_power_train:"
                + tank_type
                + ":"
                + tank_name
                + ":fuel_consumed_main_route"
            ]

        partials[
            "data:LCA:operation:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:flight_per_fu",
        ] = partial_flight_per_fu / (1.0 - leak_percentage)
        partials[
            "data:LCA:operation:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
        ] = (
            partial_flight_per_fu
            * inputs["data:environmental_impact:aircraft_per_fu"]
            / (1.0 - leak_percentage) ** 2.0
        )

        partials[
            "data:LCA:operation:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:flight_per_fu",
        ] = partial_flight_per_fu * leak_percentage / (1.0 - leak_percentage)
        partials[
            "data:LCA:operation:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
        ] = (
            partial_flight_per_fu
            * inputs["data:environmental_impact:aircraft_per_fu"]
            / (1.0 - leak_percentage) ** 2.0
        )

        partials[
            "data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * partial_flight_per_fu
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:line_test:mission_ratio",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:manufacturing:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            * inputs["data:environmental_impact:line_test:mission_ratio"]
            / (1.0 - leak_percentage) ** 2.0
            / 100.0
        )

        partials[
            "data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:environmental_impact:line_test:mission_ratio"]
            * partial_flight_per_fu
            * leak_percentage
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:line_test:mission_ratio",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            * leak_percentage
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:manufacturing:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            * inputs["data:environmental_impact:line_test:mission_ratio"]
            / (1.0 - leak_percentage) ** 2.0
            / 100.0
        )

        partials[
            "data:LCA:distribution:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * partial_flight_per_fu
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:distribution:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:delivery:mission_ratio",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:distribution:he_power_train:hydrogen:mass_per_fu",
            "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            * inputs["data:environmental_impact:delivery:mission_ratio"]
            / (1.0 - leak_percentage) ** 2.0
            / 100.0
        )

        partials[
            "data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = (
            inputs["data:environmental_impact:delivery:mission_ratio"]
            * partial_flight_per_fu
            * leak_percentage
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:delivery:mission_ratio",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            * leak_percentage
            / (1.0 - leak_percentage)
        )
        partials[
            "data:LCA:distribution:he_power_train:hydrogen:leakage_mass_per_fu",
            "data:environmental_impact:hydrogen:infrastructure_leak_percentage",
        ] = (
            inputs["data:environmental_impact:aircraft_per_fu"]
            * partial_flight_per_fu
            * inputs["data:environmental_impact:delivery:mission_ratio"]
            / (1.0 - leak_percentage) ** 2.0
            / 100.0
        )

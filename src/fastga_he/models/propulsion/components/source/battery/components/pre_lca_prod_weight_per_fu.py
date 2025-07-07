# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SERVICE_BATTERY_MASS_PER_FU

SUBMODEL_BATTERY_MASS_PER_FU_FROM_LIFESPAN = (
    "fastga_he.submodel.propulsion.battery.mass_per_fu.from_battery_lifespan"
)
SUBMODEL_BATTERY_MASS_PER_FU_FROM_HOURS_FLOWN = (
    "fastga_he.submodel.propulsion.battery.mass_per_fu.from_hours_flown"
)

oad.RegisterSubmodel.active_models[SERVICE_BATTERY_MASS_PER_FU] = (
    SUBMODEL_BATTERY_MASS_PER_FU_FROM_LIFESPAN
)


@oad.RegisterSubmodel(SERVICE_BATTERY_MASS_PER_FU, SUBMODEL_BATTERY_MASS_PER_FU_FROM_LIFESPAN)
class PreLCABatteryProdWeightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the battery pack",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            val=500.0,
            desc="Expected lifetime of the battery pack, expressed in cycles. Default value is the "
            "number of cycle required for the reference cell to reach 60% nominal capacity",
        )
        self.add_input(
            name="data:TLAR:aircraft_lifespan",
            val=np.nan,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )
        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the battery pack required for a functional unit",
        )

    def setup_partials(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            wrt=[
                "data:TLAR:aircraft_lifespan",
                "data:TLAR:flight_per_year",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu"
        ] = (
            inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * np.ceil(
                inputs["data:TLAR:aircraft_lifespan"]
                * inputs["data:TLAR:flight_per_year"]
                / inputs[
                    "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"
                ]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass"
        ] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            / inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":lifespan"]
        )


@oad.RegisterSubmodel(SERVICE_BATTERY_MASS_PER_FU, SUBMODEL_BATTERY_MASS_PER_FU_FROM_HOURS_FLOWN)
class PreLCABatteryProdWeightPerFUFromHours(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the battery pack",
        )
        self.add_input(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":hours_per_battery",
            val=250.0,
            units="h",
            desc="Expected number of hours that can be flown before the battery needs to be replace.",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            name="data:TLAR:max_airframe_hours",
            val=3524.9,
            units="h",
            desc="Expected lifetime of the aircraft expressed in airframe hours",
        )

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the battery pack required for a functional unit",
        )

    def setup_partials(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            wrt=[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":hours_per_battery",
                "data:TLAR:max_airframe_hours",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu"
        ] = (
            inputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * np.ceil(
                inputs["data:TLAR:max_airframe_hours"]
                / inputs[
                    "data:propulsion:he_power_train:battery_pack:"
                    + battery_pack_id
                    + ":hours_per_battery"
                ]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * np.ceil(
            inputs["data:TLAR:max_airframe_hours"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":hours_per_battery"
            ]
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":mass"
        ] * np.ceil(
            inputs["data:TLAR:max_airframe_hours"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":hours_per_battery"
            ]
        )

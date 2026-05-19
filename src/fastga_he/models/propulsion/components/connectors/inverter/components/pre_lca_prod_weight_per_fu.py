# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAInverterProdWeightPerFU(om.ExplicitComponent):
    """
    Computation of the weight per functional unit considering the replacement necessary
    during the lifespan of the aircraft. For the default value of the average lifespan of the
    inverter, the value is taken from :cite:`thonemann:2024` for short term technologies.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the inverter",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )
        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan",
            units="h",
            val=3.4e4,
            desc="Expected lifetime of the inverter, based on the lifespan of the IGBTs",
        )
        self.add_input(
            name="data:TLAR:aircraft_lifespan",
            val=np.nan,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Mass of the inverter required for a functional unit",
        )

    def setup_partials(self):
        inverter_id = self.options["inverter_id"]

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
                "data:environmental_impact:aircraft_per_fu",
            ],
            method="exact",
        )
        # I unfortunately have to put fd since there is no analytical expression for the
        # derivative of ceil and openmdao does not like when a nil derivative is declared
        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan",
                "data:TLAR:aircraft_lifespan",
                "data:TLAR:flight_hours_per_year",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        mass = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"]
        aircraft_per_fu = inputs["data:environmental_impact:aircraft_per_fu"]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan"]
        aircraft_lifespan = inputs["data:TLAR:aircraft_lifespan"]

        outputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu"] = (
            mass * aircraft_per_fu * np.ceil(aircraft_lifespan * flight_hours_per_year / lifespan)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        mass = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":mass"]
        aircraft_per_fu = inputs["data:environmental_impact:aircraft_per_fu"]
        flight_hours_per_year = inputs["data:TLAR:flight_hours_per_year"]
        lifespan = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":lifespan"]
        aircraft_lifespan = inputs["data:TLAR:aircraft_lifespan"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass",
        ] = mass * np.ceil(aircraft_lifespan * flight_hours_per_year / lifespan)

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = aircraft_per_fu * np.ceil(aircraft_lifespan * flight_hours_per_year / lifespan)

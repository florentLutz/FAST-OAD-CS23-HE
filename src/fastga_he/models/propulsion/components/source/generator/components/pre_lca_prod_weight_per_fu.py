# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCAGeneratorProdWeightPerFU(om.ExplicitComponent):
    """
    Computation of the weight per functional unit considering the replacement necessary
    during the lifespan of the aircraft. For the default value of the average lifespan of the
    generator, the value is taken from :cite:`thonemann:2024` for short term technologies.
    """

    def initialize(self):
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        generator_id = self.options["generator_id"]

        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":mass",
            units="kg",
            val=np.nan,
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the generator, typically around 15 year",
        )
        self.add_input(
            name="data:TLAR:aircraft_lifespan",
            val=np.nan,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )

        self.add_output(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Weight of the generator required for a functional unit",
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:environmental_impact:aircraft_per_fu",
                "data:propulsion:he_power_train:generator:" + generator_id + ":mass",
            ],
            method="exact",
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:TLAR:aircraft_lifespan",
                "data:propulsion:he_power_train:generator:" + generator_id + ":lifespan",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        outputs["data:propulsion:he_power_train:generator:" + generator_id + ":mass_per_fu"] = (
            inputs["data:propulsion:he_power_train:generator:" + generator_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * np.ceil(
                inputs["data:TLAR:aircraft_lifespan"]
                / inputs["data:propulsion:he_power_train:generator:" + generator_id + ":lifespan"]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        generator_id = self.options["generator_id"]

        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":mass_per_fu",
            "data:propulsion:he_power_train:generator:" + generator_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            / inputs["data:propulsion:he_power_train:generator:" + generator_id + ":lifespan"]
        )
        partials[
            "data:propulsion:he_power_train:generator:" + generator_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs["data:propulsion:he_power_train:generator:" + generator_id + ":mass"] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            / inputs["data:propulsion:he_power_train:generator:" + generator_id + ":lifespan"]
        )

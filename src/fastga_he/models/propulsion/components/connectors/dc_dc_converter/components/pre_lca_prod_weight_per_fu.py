# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PreLCADCDCConverterProdWeightPerFU(om.ExplicitComponent):
    """
    Computation of the weight per functional unit considering the replacement necessary
    during the lifespan of the aircraft. For the default value of the average lifespan of the
    DC/DC converter, the value is taken from :cite:`thonemann:2024` for short term technologies.
    """

    def initialize(self):
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
            units="kg",
            val=np.nan,
            desc="Mass of the converter",
        )
        self.add_input(
            name="data:environmental_impact:aircraft_per_fu",
            val=np.nan,
            desc="Number of aircraft required for a functional unit",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the DC_DC_converter, typically around 15 year",
        )
        self.add_input(
            name="data:TLAR:aircraft_lifespan",
            val=np.nan,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":mass_per_fu",
            units="kg",
            val=1e-6,
            desc="Mass of the DC_DC_converter required for a functional unit",
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
                "data:environmental_impact:aircraft_per_fu",
            ],
            method="exact",
        )
        # I unfortunately have to put fd since there is no analytical expression for the
        # derivative of ceil and openmdao does not like when a nil derivative is declared
        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":lifespan",
                "data:TLAR:aircraft_lifespan",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass_per_fu"
        ] = (
            inputs["data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"]
            * inputs["data:environmental_impact:aircraft_per_fu"]
            * np.ceil(
                inputs["data:TLAR:aircraft_lifespan"]
                / inputs[
                    "data:propulsion:he_power_train:DC_DC_converter:"
                    + dc_dc_converter_id
                    + ":lifespan"
                ]
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass_per_fu",
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass",
        ] = inputs["data:environmental_impact:aircraft_per_fu"] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass_per_fu",
            "data:environmental_impact:aircraft_per_fu",
        ] = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":mass"
        ] * np.ceil(
            inputs["data:TLAR:aircraft_lifespan"]
            / inputs[
                "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":lifespan"
            ]
        )

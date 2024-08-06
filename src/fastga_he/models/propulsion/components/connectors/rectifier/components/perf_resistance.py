# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesResistance(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:igbt",
            val=0.0041,
            units="degK**-1",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:diode",
            val=0.0033,
            units="degK**-1",
        )
        self.add_input(
            "diode_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            "IGBT_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            name="settings:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":reference_temperature",
            val=293.15,
            units="degK",
        )

        self.add_output(
            "resistance_igbt",
            val=np.full(number_of_points, 1.0e-3),
            units="ohm",
            shape=number_of_points,
        )
        self.add_output(
            "resistance_diode",
            val=np.full(number_of_points, 1.0e-3),
            units="ohm",
            shape=number_of_points,
        )

        self.declare_partials(
            of="resistance_igbt",
            wrt=[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":properties:resistance_temperature_scale_factor:igbt",
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":reference_temperature",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="resistance_igbt",
            wrt="IGBT_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="resistance_diode",
            wrt=[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
                "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":properties:resistance_temperature_scale_factor:diode",
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":reference_temperature",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="resistance_diode",
            wrt="diode_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        diode_temperature = inputs["diode_temperature"]
        igbt_temperature = inputs["IGBT_temperature"]

        rectifier_reference_temperature = inputs[
            "settings:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":reference_temperature"
        ]

        alpha_igbt = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:igbt"
        ]
        alpha_diode = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:diode"
        ]

        reference_resistance_igbt = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance"
        ]
        reference_resistance_diode = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance"
        ]

        resistance_igbt = reference_resistance_igbt * (
            1.0 + alpha_igbt * (igbt_temperature - rectifier_reference_temperature)
        )
        resistance_diode = reference_resistance_diode * (
            1.0 + alpha_diode * (diode_temperature - rectifier_reference_temperature)
        )

        outputs["resistance_igbt"] = resistance_igbt
        outputs["resistance_diode"] = resistance_diode

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        diode_temperature = inputs["diode_temperature"]
        igbt_temperature = inputs["IGBT_temperature"]

        rectifier_reference_temperature = inputs[
            "settings:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":reference_temperature"
        ]

        alpha_igbt = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:igbt"
        ]
        alpha_diode = inputs[
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:diode"
        ]

        reference_resistance_igbt = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance"
        ]
        reference_resistance_diode = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance"
        ]

        partials[
            "resistance_igbt",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
        ] = 1.0 + alpha_igbt * (igbt_temperature - rectifier_reference_temperature)
        partials[
            "resistance_igbt",
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:igbt",
        ] = reference_resistance_igbt * (igbt_temperature - rectifier_reference_temperature)
        partials["resistance_igbt", "IGBT_temperature"] = np.ones(number_of_points) * (
            reference_resistance_igbt * alpha_igbt
        )
        partials[
            "resistance_igbt",
            "settings:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":reference_temperature",
        ] = -reference_resistance_igbt * alpha_igbt

        partials[
            "resistance_diode",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
        ] = 1.0 + alpha_diode * (diode_temperature - rectifier_reference_temperature)
        partials[
            "resistance_diode",
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":properties:resistance_temperature_scale_factor:diode",
        ] = reference_resistance_diode * (diode_temperature - rectifier_reference_temperature)
        partials["resistance_diode", "diode_temperature"] = (
            np.ones(number_of_points) * alpha_diode * reference_resistance_diode
        )
        partials[
            "resistance_diode",
            "settings:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":reference_temperature",
        ] = -reference_resistance_diode * alpha_diode

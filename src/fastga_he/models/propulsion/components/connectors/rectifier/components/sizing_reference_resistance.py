# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRectifierResistances(om.ExplicitComponent):
    """
    Computation of resistances of the diodes and IGBT, reference IGBT module for this is the
    SEMiX453GB12M7p.
    """

    def initialize(self):
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

        self.options.declare(
            name="R_igbt_ref",
            types=float,
            default=1.51e-3,
            desc="Reference IGBT resistance (Ohm)",
        )
        self.options.declare(
            name="R_diode_ref",
            types=float,
            default=1.87e-3,
            desc="Reference diode resistance (Ohm)",
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance",
            val=np.nan,
        )

        self.add_input(
            "settings:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:k_resistance",
            val=1.0,
            units="unitless",
        )
        self.add_input(
            "settings:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:k_resistance",
            val=1.0,
            units="unitless",
        )

        self.add_output(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
            val=1e-3,
            units="ohm",
        )
        self.add_output(
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
            val=1e-3,
            units="ohm",
        )

    def setup_partials(self):
        rectifier_id = self.options["rectifier_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
            wrt=[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance",
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":igbt:k_resistance",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
            wrt=[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance",
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":diode:k_resistance",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance"] = (
            inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance"
            ]
            * inputs[
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":igbt:k_resistance"
            ]
            * self.options["R_igbt_ref"]
        )

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance"
        ] = (
            inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance"
            ]
            * inputs[
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":diode:k_resistance"
            ]
            * self.options["R_diode_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rectifier_id = self.options["rectifier_id"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance",
        ] = (
            self.options["R_igbt_ref"]
            * inputs[
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":igbt:k_resistance"
            ]
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:resistance",
            "settings:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:k_resistance",
        ] = (
            self.options["R_igbt_ref"]
            * inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance"
            ]
        )

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance",
        ] = (
            self.options["R_diode_ref"]
            * inputs[
                "settings:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":diode:k_resistance"
            ]
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:resistance",
            "settings:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:k_resistance",
        ] = (
            self.options["R_diode_ref"]
            * inputs[
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":scaling:resistance"
            ]
        )

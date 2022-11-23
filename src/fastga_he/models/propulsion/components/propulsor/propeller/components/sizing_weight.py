# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPropellerWeight(om.ExplicitComponent):
    """
    Computation of the weight of a propeller. Based on a regression on the data available on
    propeller certified by EASA CS-P. Assumes that the weight of blade is made up of a constant
    part, a part which depends on the volume (hence proportional to D**3) and a part proportional to
    the mechanical constraints (proportional to torque max).
    Regression can be seen in ..methodology.propeller_weight, propeller database can be seen in
    ..methodology.propeller_database.
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            val=np.nan,
            units="N*m",
            desc="Max continuous torque of the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="cm",
            desc="Diameter of the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":number_blades",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":material",
            val=1.0,
            desc="1.0 for composite, 0.0 for aluminium",
        )

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":mass",
            val=20.0,
            units="kg",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        torque_cont = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"
        ]
        prop_diameter = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"
        ]
        nb_blades = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":number_blades"
        ]
        prop_material = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":material"
        ]

        blade_weight = (9.03 + 1.074e-3 * torque_cont + 2.841e-7 * prop_diameter ** 3.0) / (
            1.0 + 0.66 * prop_material
        )

        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":mass"] = (
            blade_weight * nb_blades
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        torque_cont = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"
        ]
        prop_diameter = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"
        ]
        nb_blades = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":number_blades"
        ]
        prop_material = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":material"
        ]

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":mass",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":number_blades",
        ] = (9.03 + 1.074e-3 * torque_cont + 2.841e-7 * prop_diameter ** 3.0) / (
            1.0 + 0.66 * prop_material
        )
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":mass",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":material",
        ] = (
            -0.66
            * nb_blades
            * (9.03 + 1.074e-3 * torque_cont + 2.841e-7 * prop_diameter ** 3.0)
            / (1.0 + 0.66 * prop_material) ** 2.0
        )
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":mass",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            nb_blades * 2.841e-7 * 3.0 * prop_diameter ** 2.0 / (1.0 + 0.66 * prop_material)
        )
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":mass",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
        ] = (
            nb_blades * 1.074e-3 / (1.0 + 0.66 * prop_material)
        )

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerBlownAreaRatio(om.ExplicitComponent):
    """
    Computes the ration of the wing area blown by that propeller. Will consider the contraction
    of the slipstream with the wing AC as the reference point.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "contraction_ratio",
            val=np.nan,
            shape=number_of_points,
            desc="Contraction ratio of the propeller slipstream evaluated at the wing AC",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
            val=np.nan,
            units="m",
            desc="Value of the wing chord behind the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output(
            "blown_area_ratio",
            val=0.01,
            shape=number_of_points,
            desc="Portion of the wing blown by the propeller",
        )

        self.declare_partials(
            of="blown_area_ratio",
            wrt="contraction_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="blown_area_ratio",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
                "data:geometry:wing:area",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        wing_area = inputs["data:geometry:wing:area"]
        contraction_ratio = inputs["contraction_ratio"]
        prop_dia = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        ref_chord = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
        ]

        k_b = contraction_ratio * prop_dia * ref_chord / wing_area

        outputs["blown_area_ratio"] = k_b

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        wing_area = inputs["data:geometry:wing:area"]
        contraction_ratio = inputs["contraction_ratio"]
        prop_dia = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        ref_chord = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
        ]

        partials["blown_area_ratio", "data:geometry:wing:area"] = (
            -contraction_ratio * prop_dia * ref_chord / wing_area**2.0
        )
        partials["blown_area_ratio", "contraction_ratio"] = np.ones(number_of_points) * (
            prop_dia * ref_chord / wing_area
        )
        partials[
            "blown_area_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = contraction_ratio * ref_chord / wing_area
        partials[
            "blown_area_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
        ] = contraction_ratio * prop_dia / wing_area

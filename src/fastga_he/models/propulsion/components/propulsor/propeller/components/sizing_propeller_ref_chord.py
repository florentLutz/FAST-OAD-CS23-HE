# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import openmdao.api as om

from ..constants import POSSIBLE_POSITION

STEP = 1e-7


class SizingPropellerReferenceChord(om.ExplicitComponent):
    """
    Computation of the wing chord behind the wing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.spline = None

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the propeller, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
            val=np.nan,
            desc="Location of the propeller along the span as a fraction of the span",
        )
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )

        # This add_input is needed because in the other module, the shape of this vector is
        # copied based on the Y_vector and not having it here would cause the code to crash.
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
            val=0.0,
            units="m",
            desc="Value of the wing chord behind the propeller",
        )

        if position == "on_the_wing":
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                wrt=[
                    "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
                    "data:geometry:wing:span",
                ],
                method="exact",
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                wrt=[
                    "data:aerodynamics:wing:low_speed:Y_vector",
                    "data:aerodynamics:wing:low_speed:chord_vector",
                ],
                method="fd",
                step=STEP,
            )
        else:
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                wrt="data:aerodynamics:wing:low_speed:chord_vector",
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        if position == "on_the_wing":

            y_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio"
            ]
            half_span = inputs["data:geometry:wing:span"] / 2.0
            y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
            chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]

            idx_valid = y_vector > STEP

            self.spline = InterpolatedUnivariateSpline(
                y_vector[idx_valid], chord_vector[idx_valid], k=1
            )

            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
            ] = self.spline(half_span * y_ratio)

        else:

            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
            ] = inputs["data:aerodynamics:wing:low_speed:chord_vector"][0]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        if position == "on_the_wing":

            y_ratio = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio"
            ]
            half_span = inputs["data:geometry:wing:span"] / 2.0

            spline_value = self.spline.derivative()(half_span * y_ratio)

            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
            ] = (
                spline_value * half_span
            )
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                "data:geometry:wing:span",
            ] = (
                spline_value * y_ratio / 2.0
            )

        else:

            partial_chord = np.zeros_like(inputs["data:aerodynamics:wing:low_speed:chord_vector"])
            partial_chord[0] = 1.0
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                "data:aerodynamics:wing:low_speed:chord_vector",
            ] = partial_chord

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import openmdao.api as om

from ..constants import POSSIBLE_POSITION

STEP = 1e-7


class SizingPropellerReferenceCl(om.ExplicitComponent):
    """
    For the computation of the slipstream effects, the "clean" lift coefficient of the section
    behind the propeller will be required. As is done elsewhere is FAST-OAD_CS23 we will assume
    that the "shape" of the lift distribution won't change so to obtain the said lift coefficient
    we will simply interpolate the Cl=f(y) curve.

    As a first step, the slipstream effects will only be available if the propeller is on the
    wing, hence why the returned cl is equal to zero elsewhere.
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

        # We leave those inputs here in case there is a weird bug with the copying of shapes
        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
            val=0.0,
            desc="Value of the clean lift coefficient of the section behind the propeller for "
            "reference wing lift coefficient",
        )

        if position == "on_the_wing":

            self.add_input(
                name="data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
                val=np.nan,
                desc="Location of the propeller along the span as a fraction of the span",
            )
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")

            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
                wrt=[
                    "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
                    "data:geometry:wing:span",
                ],
                method="exact",
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
                wrt=[
                    "data:aerodynamics:wing:low_speed:Y_vector",
                    "data:aerodynamics:wing:low_speed:CL_vector",
                ],
                method="fd",
                step=STEP,
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
            cl_vector = inputs["data:aerodynamics:wing:low_speed:CL_vector"]

            idx_valid = y_vector > STEP

            self.spline = InterpolatedUnivariateSpline(
                y_vector[idx_valid], cl_vector[idx_valid], k=2
            )

            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref"
            ] = self.spline(half_span * y_ratio)

        else:

            # "Cheap" way to turn off slipstream effects in case the propeller is not on the wing
            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref"
            ] = 0.0

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
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
            ] = (
                spline_value * half_span
            )
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
                "data:geometry:wing:span",
            ] = (
                spline_value * y_ratio / 2.0
            )

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
sizing_ducted_fan_ref_chord.py
================================

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import openmdao.api as om

STEP = 1e-7


class SizingDuctedFanReferenceChord(om.ExplicitComponent):
    """
    Computation of the wing chord at the spanwise station of the ducted fan.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.spline = None

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]

        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.nan,
            units="m",
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )
        # Needed only so the shape of chord_vector/Y_vector resolves consistently wherever this
        # trio is connected together elsewhere in the model (same reasoning/comment as in the
        # propeller's original component).
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )
        self.add_input(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio",
            val=np.nan,
            desc="Location of the ducted fan along the span as a fraction of the span",
        )
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
            val=0.0,
            units="m",
            desc="Value of the wing chord at the spanwise station of the ducted fan",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
            wrt=[
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio",
                "data:geometry:wing:span",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
            wrt=[
                "data:aerodynamics:wing:low_speed:Y_vector",
                "data:aerodynamics:wing:low_speed:chord_vector",
            ],
            method="fd",
            step=STEP,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        y_ratio = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio"
        ]
        half_span = inputs["data:geometry:wing:span"] / 2.0
        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]

        idx_valid = y_vector > STEP

        self.spline = InterpolatedUnivariateSpline(
            y_vector[idx_valid], chord_vector[idx_valid], k=1
        )

        outputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref"
        ] = self.spline(half_span * y_ratio)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        y_ratio = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio"
        ]
        half_span = inputs["data:geometry:wing:span"] / 2.0

        spline_value = self.spline.derivative()(half_span * y_ratio)

        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio",
        ] = spline_value * half_span
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
            "data:geometry:wing:span",
        ] = spline_value * y_ratio / 2.0

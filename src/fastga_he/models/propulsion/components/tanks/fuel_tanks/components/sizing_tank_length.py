# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import openmdao.api as om

from ..constants import POSSIBLE_POSITION

STEP = 1e-7
TE_CHORD_MARGIN_RATIO = 0.05  # Ratio of the chord left between the flaps and the tank


class SizingFuelTankLength(om.ExplicitComponent):
    """
    Computation of the reference length for the computation of the tank width. If the tank is in
    a pod, it will depend on volume and a fineness ratio. If it is in the wing, it will depend on
    the wing chord law, its position and some factor. If it is in the fuselage it will be equal
    to the cabin length.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.spline = None

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the fuel tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
            val=0.5,
            units="m",
            desc="Value of the length of the tank in the x-direction, computed differently based "
            "on the location of the tank",
        )

        if position == "inside_the_wing":
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

            self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y",
                units="m",
                val=np.nan,
                desc="Y position of the fuel tank center of gravity",
            )

            self.declare_partials(
                of="data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
                wrt=[
                    "data:geometry:flap:chord_ratio",
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y",
                ],
                method="exact",
            )

            self.add_output(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord",
                val=0.0,
                units="m",
                desc="Reference wing chord for the tank",
            )

            self.declare_partials(
                of="data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:ref_chord",
                wrt=[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y",
                ],
                method="exact",
            )

        elif position == "wing_pod":
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
                units="m**3",
                val=np.nan,
                desc="Capacity of the tank in terms of volume",
            )
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio",
                val=15.0,
                desc="Ratio between the wing pod length and width/height ",
            )

            self.declare_partials(of="*", wrt="*", method="exact")

        else:
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

            self.declare_partials(of="*", wrt="data:geometry:cabin:length", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            y_position = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y"
            ]
            y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
            chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]
            flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]

            idx_valid = y_vector > STEP

            self.spline = InterpolatedUnivariateSpline(
                y_vector[idx_valid], chord_vector[idx_valid], k=1
            )

            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord"
            ] = self.spline(y_position)
            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length"
            ] = self.spline(y_position) * (1.0 - TE_CHORD_MARGIN_RATIO - flap_chord_ratio)

        elif position == "wing_pod":
            fineness_ratio = inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio"
            ]
            tank_volume = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"
            ]

            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length"
            ] = (tank_volume * fineness_ratio**2.0) ** (1.0 / 3.0)

        else:
            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length"
            ] = inputs["data:geometry:cabin:length"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            y_position = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y"
            ]
            flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]

            spline_derivative_value = self.spline.derivative()(y_position)
            spline_value = self.spline(y_position)

            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y",
            ] = spline_derivative_value * (1.0 - flap_chord_ratio - TE_CHORD_MARGIN_RATIO)
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
                "data:geometry:flap:chord_ratio",
            ] = -spline_value

            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":CG:y",
            ] = spline_derivative_value

        elif position == "wing_pod":
            fineness_ratio = inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio"
            ]
            tank_volume = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"
            ]

            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio",
            ] = (
                1.0
                / 3.0
                * (tank_volume * fineness_ratio**2.0) ** (-2.0 / 3.0)
                * 2.0
                * fineness_ratio
                * tank_volume
            )
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            ] = (
                1.0
                / 3.0
                * (tank_volume * fineness_ratio**2.0) ** (-2.0 / 3.0)
                * fineness_ratio**2.0
            )

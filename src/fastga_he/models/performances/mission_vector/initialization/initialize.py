# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.
import numpy as np
import openmdao.api as om

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
    POINTS_NB_DESCENT,
)
from ..initialization.initialize_cg import InitializeCoG
from ..initialization.initialize_airspeed import InitializeAirspeed
from ..initialization.initialize_airspeed_derivatives import InitializeAirspeedDerivatives
from ..initialization.initialize_altitude import InitializeAltitude
from ..initialization.initialize_gamma import InitializeGamma
from ..initialization.initialize_horizontal_speed import InitializeHorizontalSpeed
from ..initialization.initialize_time_and_distance import InitializeTimeAndDistance


class Initialize(om.Group):
    """Find the conditions necessary for the aircraft equilibrium."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        engine_setting = np.concatenate(
            (
                np.full(POINTS_NB_CLIMB, 2),
                np.full(POINTS_NB_CRUISE, 3),
                np.full(POINTS_NB_DESCENT, 2),
            )
        )
        ivc_engine_setting = om.IndepVarComp()
        ivc_engine_setting.add_output(
            "engine_setting", val=engine_setting, units=None, shape=number_of_points
        )

        self.add_subsystem("initialize_engine_setting", subsys=ivc_engine_setting, promotes=[])
        self.add_subsystem(
            "initialize_altitude",
            InitializeAltitude(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "initialize_airspeed",
            InitializeAirspeed(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "initialize_gamma",
            InitializeGamma(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "initialize_horizontal_speed",
            InitializeHorizontalSpeed(number_of_points=number_of_points),
            promotes_inputs=[],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "initialize_time_and_distance",
            InitializeTimeAndDistance(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "initialize_airspeed_time_derivatives",
            InitializeAirspeedDerivatives(number_of_points=number_of_points),
            promotes_inputs=[],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "initialize_center_of_gravity",
            InitializeCoG(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )

        self.connect(
            "initialize_horizontal_speed.horizontal_speed",
            "initialize_time_and_distance.horizontal_speed",
        )

        self.connect(
            "initialize_airspeed.true_airspeed",
            [
                "initialize_gamma.true_airspeed",
                "initialize_horizontal_speed.true_airspeed",
                "initialize_time_and_distance.true_airspeed",
                "initialize_airspeed_time_derivatives.true_airspeed",
            ],
        )

        self.connect(
            "initialize_airspeed.equivalent_airspeed",
            "initialize_airspeed_time_derivatives.equivalent_airspeed",
        )

        self.connect(
            "initialize_gamma.gamma",
            ["initialize_airspeed_time_derivatives.gamma", "initialize_horizontal_speed.gamma"],
        )

        self.connect(
            "initialize_altitude.altitude",
            [
                "initialize_airspeed.altitude",
                "initialize_gamma.altitude",
                "initialize_time_and_distance.altitude",
                "initialize_airspeed_time_derivatives.altitude",
            ],
        )

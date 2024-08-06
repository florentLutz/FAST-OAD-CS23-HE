# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere

MIN_POWER_TAXI = 5e3


class ThrustTaxi(om.ExplicitComponent):
    """Computes the fuel consumed during the taxi phases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.below_min_thrust_ti = None
        self.below_min_thrust_to = None

    def setup(self):
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", np.nan)

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_out:thrust", 500, units="N")

        self.declare_partials(
            of="data:mission:sizing:taxi_out:thrust",
            wrt=[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:wing:low_speed:CL0_clean",
                "data:aerodynamics:wing:low_speed:induced_drag_coefficient",
                "data:geometry:wing:area",
                "data:mission:sizing:taxi_out:speed",
            ],
        )

        self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_in:thrust", 500, units="N")

        self.declare_partials(
            of="data:mission:sizing:taxi_in:thrust",
            wrt=[
                "data:aerodynamics:aircraft:low_speed:CD0",
                "data:aerodynamics:wing:low_speed:CL0_clean",
                "data:aerodynamics:wing:low_speed:induced_drag_coefficient",
                "data:geometry:wing:area",
                "data:mission:sizing:taxi_in:speed",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
        coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]

        wing_area = inputs["data:geometry:wing:area"]

        speed_to = inputs["data:mission:sizing:taxi_out:speed"]
        speed_ti = inputs["data:mission:sizing:taxi_in:speed"]

        cd = cd0 + coeff_k_wing * cl0_wing**2.0
        density = Atmosphere(altitude=0.0).density

        # To avoid issue with the convergence of the power train, we will ensure that the thrust
        # required is high enough, by requesting a minimum of 5 kW during taxi
        min_thrust_to = MIN_POWER_TAXI / speed_to
        min_thrust_ti = MIN_POWER_TAXI / speed_ti

        outputs["data:mission:sizing:taxi_out:thrust"] = np.maximum(
            0.5 * density * speed_to**2.0 * wing_area * cd, min_thrust_to
        )
        self.below_min_thrust_to = (
            0 if outputs["data:mission:sizing:taxi_out:thrust"] == min_thrust_to else 1
        )
        outputs["data:mission:sizing:taxi_in:thrust"] = np.maximum(
            0.5 * density * speed_ti**2.0 * wing_area * cd, min_thrust_ti
        )
        self.below_min_thrust_ti = (
            0 if outputs["data:mission:sizing:taxi_in:thrust"] == min_thrust_ti else 1
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
        coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]

        wing_area = inputs["data:geometry:wing:area"]

        speed_to = inputs["data:mission:sizing:taxi_out:speed"]
        speed_ti = inputs["data:mission:sizing:taxi_in:speed"]

        cd = cd0 + coeff_k_wing * cl0_wing**2.0
        density = Atmosphere(altitude=0.0).density

        # Taxi-out
        partials[
            "data:mission:sizing:taxi_out:thrust",
            "data:aerodynamics:aircraft:low_speed:CD0",
        ] = 0.5 * density * speed_to**2.0 * wing_area * self.below_min_thrust_to
        partials[
            "data:mission:sizing:taxi_out:thrust",
            "data:aerodynamics:wing:low_speed:CL0_clean",
        ] = density * speed_to**2.0 * wing_area * coeff_k_wing * cl0_wing * self.below_min_thrust_to
        partials[
            "data:mission:sizing:taxi_out:thrust",
            "data:aerodynamics:wing:low_speed:induced_drag_coefficient",
        ] = 0.5 * density * speed_to**2.0 * wing_area * cl0_wing**2.0 * self.below_min_thrust_to
        partials[
            "data:mission:sizing:taxi_out:thrust",
            "data:geometry:wing:area",
        ] = 0.5 * density * speed_to**2.0 * cd * self.below_min_thrust_to
        partials[
            "data:mission:sizing:taxi_out:thrust",
            "data:mission:sizing:taxi_out:speed",
        ] = (
            (density * speed_to * wing_area * cd * self.below_min_thrust_to)
            if self.below_min_thrust_to
            else -MIN_POWER_TAXI / speed_to**2.0
        )

        # Taxi-in
        partials[
            "data:mission:sizing:taxi_in:thrust",
            "data:aerodynamics:aircraft:low_speed:CD0",
        ] = 0.5 * density * speed_ti**2.0 * wing_area * self.below_min_thrust_ti
        partials[
            "data:mission:sizing:taxi_in:thrust",
            "data:aerodynamics:wing:low_speed:CL0_clean",
        ] = density * speed_ti**2.0 * wing_area * coeff_k_wing * cl0_wing * self.below_min_thrust_ti
        partials[
            "data:mission:sizing:taxi_in:thrust",
            "data:aerodynamics:wing:low_speed:induced_drag_coefficient",
        ] = 0.5 * density * speed_ti**2.0 * wing_area * cl0_wing**2.0 * self.below_min_thrust_ti
        partials[
            "data:mission:sizing:taxi_in:thrust",
            "data:geometry:wing:area",
        ] = 0.5 * density * speed_ti**2.0 * cd * self.below_min_thrust_ti
        partials[
            "data:mission:sizing:taxi_in:thrust",
            "data:mission:sizing:taxi_in:speed",
        ] = (
            (density * speed_ti * wing_area * cd * self.below_min_thrust_ti)
            if self.below_min_thrust_to
            else -MIN_POWER_TAXI / speed_ti**2.0
        )

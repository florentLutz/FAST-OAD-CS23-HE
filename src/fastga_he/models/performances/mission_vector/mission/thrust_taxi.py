# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

from stdatm import Atmosphere


class ThrustTaxi(om.ExplicitComponent):
    """Computes the fuel consumed during the taxi phases."""

    def setup(self):

        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", np.nan)

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_out:thrust", 1500, units="N")

        self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_in:thrust", 1500, units="N")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
        coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]

        wing_area = inputs["data:geometry:wing:area"]

        speed_to = inputs["data:mission:sizing:taxi_out:speed"]
        speed_ti = inputs["data:mission:sizing:taxi_in:speed"]

        cd = cd0 + coeff_k_wing * cl0_wing ** 2.0
        density = Atmosphere(altitude=0.0).density

        outputs["data:mission:sizing:taxi_out:thrust"] = (
            0.5 * density * speed_to ** 2.0 * wing_area * cd
        )
        outputs["data:mission:sizing:taxi_in:thrust"] = (
            0.5 * density * speed_ti ** 2.0 * wing_area * cd
        )

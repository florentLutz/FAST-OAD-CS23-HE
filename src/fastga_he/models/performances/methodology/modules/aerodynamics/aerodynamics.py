import numpy as np
import scipy.constants as sc

import openmdao.api as om

from fastoad.model_base import Atmosphere


class ComputeAerodynamics(om.Group):
    def setup(self):

        self.add_subsystem(name="compute_profile_drag", subsys=ComputeProfileDrag(), promotes=["*"])
        self.add_subsystem(
            name="compute_induced_drag", subsys=ComputeInducedDragCoefficient(), promotes=["*"]
        )
        self.add_subsystem(
            name="compute_lift_to_drag", subsys=ComputeLiftToDragRatio(), promotes=["*"]
        )


class ComputeInducedDragCoefficient(om.ExplicitComponent):
    """
    Computes the induced drag coefficient based on the aspect ratio
    """

    def setup(self):

        # Defining the input(s)

        self.add_input(name="aspect_ratio", val=np.nan)

        # Defining the output(s)

        self.add_output(name="induced_drag_coefficient")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Assigning the input to local variable for clarity
        aspect_ratio = inputs["aspect_ratio"]

        # Computation of the Oswald efficiency factor
        e = 1.78 * (1.0 - 0.045 * aspect_ratio ** 0.68) - 0.64

        # Computation of the lift induced drag coefficient
        k = 1.0 / (np.pi * aspect_ratio * e)

        outputs["induced_drag_coefficient"] = k


class ComputeProfileDrag(om.ExplicitComponent):
    """
    Computes the profile drag of the aircraft based on the wing area
    """

    def setup(self):

        # Defining the input(s)

        self.add_input(name="wing_area", units="m**2", val=np.nan)

        # Defining the output(s)

        self.add_output(name="profile_drag_coefficient")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Assigning the input to local variable for clarity
        wing_area = inputs["wing_area"]

        # Wet area of the aircraft without the wings
        wing_area_ref = 13.50

        # Profile drag coefficient of the aircraft without the wings
        cd0_other = 0.022

        # Constant linking the wing profile drag to its wet area, and by extension, its reference area
        c = 0.0004

        # Computation of the profile drag
        cd0 = cd0_other * wing_area_ref / wing_area + c

        outputs["profile_drag_coefficient"] = cd0


class ComputeLiftToDragRatio(om.ExplicitComponent):
    """
    Computes the lift to drag ratio considering a lift equilibrium in cruise and a simple
    quadratic model.
    """

    def setup(self):

        # Defining the input(s)

        self.add_input(
            name="cruise_altitude", units="m", val=np.nan
        )  # For a float or int, shape don't have to be
        # provided
        self.add_input(name="cruise_speed", units="m/s", val=np.nan)
        self.add_input(
            name="profile_drag_coefficient", val=np.nan
        )  # When the quantity does not have a unit, the units
        # field doesn't need to be filled
        self.add_input(name="induced_drag_coefficient", val=np.nan)
        self.add_input(name="mtow", units="kg", val=np.nan)
        self.add_input(
            name="wing_area", units="m**2", val=np.nan
        )  # OpenMDAO understands the multiplication/division
        # of units in between them hence why the m**2 is understood

        # Defining the output(s)

        self.add_output(name="l_d_ratio")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Assigning the input to local variable for clarity
        cruise_altitude = inputs["cruise_altitude"]
        cruise_speed = inputs["cruise_speed"]
        cd0 = inputs["profile_drag_coefficient"]
        k = inputs["induced_drag_coefficient"]
        mtow = inputs["mtow"]
        wing_area = inputs["wing_area"]

        # Air density at sea-level, to compute it, we will use the Atmosphere
        # model available in FAST-OAD, so we will create an Atmosphere instance
        # using the cruise altitude and extract its density attribute
        atm = Atmosphere(altitude=cruise_altitude, altitude_in_feet=False)
        rho = atm.density

        # Computation of the cruise lift coefficient using a simple equilibrium
        cl = (mtow * sc.g) / (0.5 * rho * cruise_speed ** 2.0 * wing_area)

        # Computation of the cruise drag coefficient using the simple quadratic model
        cd = cd0 + k * cl ** 2

        # Computation of the ratio
        l_d = cl / cd

        outputs["l_d_ratio"] = l_d

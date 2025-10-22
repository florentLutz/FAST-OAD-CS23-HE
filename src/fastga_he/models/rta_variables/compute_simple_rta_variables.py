# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from stdatm import AtmosphereWithPartials


class ComputeRTAVariable(om.ExplicitComponent):
    """
    Simple computation to obtain FAST-GA-HE Performances inputs from RTA outputs and set
    variables that doesn't exist in RTA or requires complex computation.
    """

    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)
        self.add_input("data:TLAR:cruise_mach", val=np.nan)
        self.add_input("data:TLAR:approach_speed", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:taxi_in:distance", val=np.nan, units="m")
        self.add_input("data:mission:sizing:taxi_in:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_out:distance", val=np.nan, units="m")
        self.add_input("data:mission:sizing:taxi_out:duration", val=np.nan, units="s")

        self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
        self.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="1/rad")
        self.add_output("data:aerodynamics:wing:cruise:CM0_clean")
        self.add_output("data:TLAR:luggage_mass_design", units="kg")
        self.add_output("data:aerodynamics:cruise:unit_reynolds", units="1/m")
        self.add_output("data:aerodynamics:low_speed:unit_reynolds", units="1/m")
        self.add_output("data:TLAR:v_cruise", units="m/s")
        self.add_output("data:mission:sizing:taxi_in:speed", units="m/s")
        self.add_output("data:mission:sizing:taxi_out:speed", units="m/s")

    def setup_partials(self):
        self.declare_partials(
            of="data:TLAR:luggage_mass_design", wrt="data:TLAR:NPAX_design", val=20
        )
        self.declare_partials(
            of="data:aerodynamics:low_speed:unit_reynolds",
            wrt="data:TLAR:approach_speed",
            method="exact",
        )
        self.declare_partials(
            of=["data:TLAR:v_cruise", "data:aerodynamics:cruise:unit_reynolds"],
            wrt=["data:mission:sizing:main_route:cruise:altitude", "data:TLAR:cruise_mach"],
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_in:speed",
            wrt=["data:mission:sizing:taxi_in:distance", "data:mission:sizing:taxi_in:duration"],
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_out:speed",
            wrt=["data:mission:sizing:taxi_out:distance", "data:mission:sizing:taxi_out:duration"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        atm_cruise = AtmosphereWithPartials(cruise_alt, altitude_in_feet=False)
        atm_cruise.mach = inputs["data:TLAR:cruise_mach"]

        atm_approach = AtmosphereWithPartials(0.0, altitude_in_feet=False)
        atm_approach.true_airspeed = inputs["data:TLAR:approach_speed"]

        outputs["data:TLAR:v_cruise"] = atm_cruise.true_airspeed

        outputs["data:aerodynamics:cruise:unit_reynolds"] = atm_cruise.unitary_reynolds

        outputs["data:aerodynamics:low_speed:unit_reynolds"] = atm_approach.unitary_reynolds

        outputs["data:TLAR:luggage_mass_design"] = 20.0 * inputs["data:TLAR:NPAX_design"]

        outputs["data:mission:sizing:taxi_in:speed"] = (
            inputs["data:mission:sizing:taxi_in:distance"]
            / inputs["data:mission:sizing:taxi_in:duration"]
        )

        outputs["data:mission:sizing:taxi_out:speed"] = (
            inputs["data:mission:sizing:taxi_out:distance"]
            / inputs["data:mission:sizing:taxi_out:duration"]
        )

        outputs["data:aerodynamics:horizontal_tail:cruise:CL0"] = -0.0068437669175491515

        outputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"] = 6.28

        outputs["data:aerodynamics:wing:cruise:CM0_clean"] = -0.02413516654351498

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        atm_cruise = AtmosphereWithPartials(cruise_alt, altitude_in_feet=False)

        atm_approach = AtmosphereWithPartials(0.0, altitude_in_feet=False)

        partials["data:mission:sizing:taxi_in:speed", "data:mission:sizing:taxi_in:distance"] = (
            1.0 / inputs["data:mission:sizing:taxi_in:duration"]
        )

        partials["data:mission:sizing:taxi_in:speed", "data:mission:sizing:taxi_in:duration"] = -(
            inputs["data:mission:sizing:taxi_in:distance"]
        ) / (inputs["data:mission:sizing:taxi_in:duration"] ** 2.0)

        partials["data:mission:sizing:taxi_out:speed", "data:mission:sizing:taxi_out:distance"] = (
            1.0 / inputs["data:mission:sizing:taxi_out:duration"]
        )

        partials["data:mission:sizing:taxi_out:speed", "data:mission:sizing:taxi_out:duration"] = -(
            inputs["data:mission:sizing:taxi_out:distance"]
        ) / (inputs["data:mission:sizing:taxi_out:duration"] ** 2.0)

        partials["data:aerodynamics:low_speed:unit_reynolds", "data:TLAR:approach_speed"] = (
            1.0 / atm_approach.kinematic_viscosity
        )

        partials["data:TLAR:v_cruise", "data:TLAR:cruise_mach"] = atm_cruise.speed_of_sound

        partials["data:TLAR:v_cruise", "data:mission:sizing:main_route:cruise:altitude"] = (
            inputs["data:TLAR:cruise_mach"] * atm_cruise.partial_speed_of_sound_altitude
        )

        partials["data:aerodynamics:cruise:unit_reynolds", "data:TLAR:cruise_mach"] = (
            atm_cruise.speed_of_sound
        ) / atm_cruise.kinematic_viscosity

        partials[
            "data:aerodynamics:cruise:unit_reynolds",
            "data:mission:sizing:main_route:cruise:altitude",
        ] = inputs["data:TLAR:cruise_mach"] * (
            (
                atm_cruise.partial_speed_of_sound_altitude * atm_cruise.kinematic_viscosity
                - atm_cruise.speed_of_sound * atm_cruise.partial_kinematic_viscosity_altitude
            )
            / atm_cruise.kinematic_viscosity**2.0
        )

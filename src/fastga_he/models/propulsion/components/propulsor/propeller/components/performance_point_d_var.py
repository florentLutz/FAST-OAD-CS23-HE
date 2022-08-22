# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from scipy.optimize import fsolve
from stdatm.atmosphere import Atmosphere

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar
from fastga.models.aerodynamics.external.propeller_code.propeller_core import PropellerCoreModule


class ComputePropellerPointPerformanceDVar(om.Group):

    """Computes propeller profiles aerodynamic coefficient and propeller behaviour."""

    def initialize(self):
        self.options.declare(
            "sections_profile_position_list",
            default=[0.0, 0.25, 0.28, 0.35, 0.40, 0.45],
            types=list,
        )
        self.options.declare(
            "sections_profile_name_list",
            default=["naca4430", "naca4424", "naca4420", "naca4414", "naca4412", "naca4409"],
            types=list,
        )
        self.options.declare("elements_number", default=20, types=int)

    def setup(self):
        self.add_subsystem("diameter_variation", _PreparePropellerDVariation(), promotes=["*"])
        self.add_subsystem("ref_twist_adjust", _PreparePropellerTwist(), promotes=["*"])
        ivc = om.IndepVarComp()
        ivc.add_output("data:aerodynamics:propeller:mach", val=0.0)
        ivc.add_output("data:aerodynamics:propeller:reynolds", val=1e6)
        self.add_subsystem("propeller_point_perf_aero_conditions", ivc, promotes=["*"])
        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar_efficiency",
                XfoilPolar(
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.connect(
                "data:aerodynamics:propeller:mach",
                profile + "_polar_efficiency.xfoil:mach",
            )
            self.connect(
                "data:aerodynamics:propeller:reynolds",
                profile + "_polar_efficiency.xfoil:reynolds",
            )
        self.add_subsystem(
            "propeller_point_perf",
            _ComputePropellerPointPerformance(
                sections_profile_position_list=self.options["sections_profile_position_list"],
                sections_profile_name_list=self.options["sections_profile_name_list"],
                elements_number=self.options["elements_number"],
            ),
            promotes_inputs=["data:*"],
            promotes_outputs=["*"],
        )

        for profile in self.options["sections_profile_name_list"]:
            self.connect(
                profile + "_polar_efficiency.xfoil:alpha",
                "propeller_point_perf." + profile + "_polar:alpha",
            )
            self.connect(
                profile + "_polar_efficiency.xfoil:CL",
                "propeller_point_perf." + profile + "_polar:CL",
            )
            self.connect(
                profile + "_polar_efficiency.xfoil:CD",
                "propeller_point_perf." + profile + "_polar:CD",
            )


class _PreparePropellerDVariation(om.ExplicitComponent):
    def setup(self):
        self.add_input(
            "data:geometry:propeller:chord_to_diameter_vect",
            val=np.nan,
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")

        self.add_output(
            "data:geometry:propeller:chord_vect",
            val=np.nan,
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:chord_to_diameter_vect",
        )
        self.add_output("data:geometry:propeller:hub_diameter", val=np.nan, units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:geometry:propeller:hub_diameter"] = (
            0.2 * inputs["data:geometry:propeller:diameter"]
        )
        outputs["data:geometry:propeller:chord_vect"] = (
            inputs["data:geometry:propeller:diameter"]
            * inputs["data:geometry:propeller:chord_to_diameter_vect"]
        )


class _PreparePropellerTwist(om.ExplicitComponent):
    def setup(self):
        self.add_input(
            "data:geometry:propeller:twist_vect_ref",
            val=np.nan,
            units="rad",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )
        self.add_input("data:geometry:propeller:radius_ratio_vect", val=np.nan, shape_by_conn=True)
        self.add_input(
            "data:aerodynamics:propeller:point_performance:twist_75_ref", units="rad", val=np.nan
        )

        self.add_output(
            "data:geometry:propeller:twist_vect",
            val=np.nan,
            units="rad",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        twist_75_ref = inputs["data:aerodynamics:propeller:point_performance:twist_75_ref"]
        twist_vect_ref = inputs["data:geometry:propeller:twist_vect_ref"]
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]

        theta_75_orig = np.interp(0.75, radius_ratio_vect, twist_vect_ref)
        twist_vect = twist_vect_ref + (twist_75_ref - theta_75_orig)

        outputs["data:geometry:propeller:twist_vect"] = twist_vect


class _ComputePropellerPointPerformance(PropellerCoreModule):
    def setup(self):

        super().setup()

        self.add_input(
            "data:aerodynamics:propeller:point_performance:twist_75", units="deg", val=np.nan
        )
        self.add_input(
            "data:aerodynamics:propeller:point_performance:rho", units="kg/m**3", val=np.nan
        )
        self.add_input(
            "data:aerodynamics:propeller:point_performance:speed", units="m/s", val=np.nan
        )

        self.add_output(
            "data:aerodynamics:propeller:point_performance:thrust", units="N", val=np.nan
        )
        self.add_output("data:aerodynamics:propeller:point_performance:efficiency", val=np.nan)
        self.add_output(
            "data:aerodynamics:propeller:point_performance:power", val=np.nan, units="W"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Define init values
        omega = inputs["data:geometry:propeller:average_rpm"]
        density = inputs["data:aerodynamics:propeller:point_performance:rho"]
        altitude = self.find_altitude(density)
        speed = inputs["data:aerodynamics:propeller:point_performance:speed"]
        theta_75 = inputs["data:aerodynamics:propeller:point_performance:twist_75"]

        prop_diameter = inputs["data:geometry:propeller:diameter"]
        radius_min = inputs["data:geometry:propeller:hub_diameter"] / 2.0
        radius_max = prop_diameter / 2.0
        length = radius_max - radius_min
        elements_number = np.arange(self.options["elements_number"])
        element_length = length / self.options["elements_number"]
        radius = radius_min + (elements_number + 0.5) * element_length
        sections_profile_position_list = self.options["sections_profile_position_list"]
        sections_profile_name_list = self.options["sections_profile_name_list"]

        # Build table with aerodynamic coefficients for quicker computations down the line
        alpha_interp = np.array([0])

        for profile in self.options["sections_profile_name_list"]:
            alpha_interp = np.union1d(alpha_interp, inputs[profile + "_polar:alpha"])

        alpha_list = np.zeros((len(radius), len(alpha_interp)))
        cl_list = np.zeros((len(radius), len(alpha_interp)))
        cd_list = np.zeros((len(radius), len(alpha_interp)))

        for idx, _ in enumerate(radius):

            index = np.where(sections_profile_position_list < (radius[idx] / radius_max))[0]
            if index is None:
                profile_name = sections_profile_name_list[0]
            else:
                profile_name = sections_profile_name_list[int(index[-1])]

            # Load profile polars
            alpha_element, cl_element, cd_element = self.reshape_polar(
                inputs[profile_name + "_polar:alpha"],
                inputs[profile_name + "_polar:CL"],
                inputs[profile_name + "_polar:CD"],
            )

            alpha_list[idx, :] = alpha_interp
            cl_list[idx, :] = np.interp(alpha_interp, alpha_element, cl_element)
            cd_list[idx, :] = np.interp(alpha_interp, alpha_element, cd_element)

        thrust, eta, torque = self.compute_pitch_performance(
            inputs, theta_75, speed, altitude, omega, radius, alpha_list, cl_list, cd_list
        )
        power = torque * omega

        outputs["data:aerodynamics:propeller:point_performance:thrust"] = thrust
        outputs["data:aerodynamics:propeller:point_performance:efficiency"] = eta
        outputs["data:aerodynamics:propeller:point_performance:power"] = power

    def find_altitude(self, density):

        alt = fsolve(self.diff_to_density, x0=np.array([0.0]), args=density)
        return max(float(alt[0]), 0.0)

    @staticmethod
    def diff_to_density(altitude, target_density):

        return Atmosphere(altitude, altitude_in_feet=False).density - target_density

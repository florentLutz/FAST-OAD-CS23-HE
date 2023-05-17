# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from scipy.optimize import fsolve
from stdatm.atmosphere import Atmosphere

from fastga.models.aerodynamics.external.propeller_code.propeller_core import PropellerCoreModule
from fastga.models.aerodynamics.constants import POLAR_POINT_COUNT
from fastga_he.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolarMod


class ComputePropellerPointPerformance(om.Group):

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
        self.add_subsystem("rescale_twist", _AdjustTwistLaw(), promotes=["*"])
        self.add_subsystem("ref_twist_adjust", _PreparePropellerTwist(), promotes=["*"])
        ivc = om.IndepVarComp()
        ivc.add_output("data:aerodynamics:propeller:mach", val=0.0)
        ivc.add_output("data:aerodynamics:propeller:reynolds", val=1e6)
        self.add_subsystem("propeller_point_perf_aero_conditions", ivc, promotes=["*"])
        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar_efficiency",
                XfoilPolarMod(
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.add_subsystem(
                profile + "_polar_efficiency_inv",
                XfoilPolarMod(
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
                    inviscid=True,
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
            self.connect(
                "data:aerodynamics:propeller:mach",
                profile + "_polar_efficiency_inv.xfoil:mach",
            )
            self.connect(
                "data:aerodynamics:propeller:reynolds",
                profile + "_polar_efficiency_inv.xfoil:reynolds",
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
                profile + "_polar_efficiency_inv.xfoil:CL",
                "propeller_point_perf." + profile + "_polar:CL_inv",
            )
            self.connect(
                profile + "_polar_efficiency.xfoil:CD",
                "propeller_point_perf." + profile + "_polar:CD",
            )
            self.connect(
                profile + "_polar_efficiency.xfoil:CD_min_2D",
                "propeller_point_perf." + profile + "_polar:CD_min_2D",
            )


class _AdjustTwistLaw(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:propeller:radius_ratio_vect", val=np.nan, shape_by_conn=True)
        self.add_input(
            "data:geometry:propeller:twist_tip",
            val=np.nan,
            desc="twist between the root section and tip section. Shape of twist law is "
            "considered constant but its amplitude may vary",
            units="deg",
        )

        self.add_output(
            "data:geometry:propeller:twist_vect_ref",
            val=np.nan,
            units="deg",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        radius_ratio_ref = np.array(
            [0.0, 0.197, 0.297, 0.400, 0.584, 0.597, 0.708, 0.803, 0.900, 0.949, 0.99]
        )
        twist_vect_ref = np.array(
            [
                34.01214457,
                34.01214457,
                31.33590222,
                27.3504266,
                19.627439,
                19.13878771,
                16.02529171,
                14.02445889,
                12.36807196,
                11.38711638,
                11.36709688,
            ]
        )
        amplitude_ref = np.max(twist_vect_ref) - np.min(twist_vect_ref)
        amplitude_new = inputs["data:geometry:propeller:twist_tip"]
        radius_ratio = inputs["data:geometry:propeller:radius_ratio_vect"]
        twist_vect = (
            np.interp(radius_ratio, radius_ratio_ref, twist_vect_ref)
            * amplitude_new
            / amplitude_ref
        )

        outputs["data:geometry:propeller:twist_vect_ref"] = twist_vect


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

        for profile in self.options["sections_profile_name_list"]:
            self.add_input(profile + "_polar:CL_inv", val=np.nan, shape=POLAR_POINT_COUNT)
            self.add_input(profile + "_polar:CD_min_2D", val=np.nan)

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
        cl_inv_list = np.zeros((len(radius), len(alpha_interp)))
        cd_list = np.zeros((len(radius), len(alpha_interp)))
        cd_min_list = np.zeros(len(radius))

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
            _, cl_inv_element, _ = self.reshape_polar(
                inputs[profile_name + "_polar:alpha"],
                inputs[profile_name + "_polar:CL_inv"],
                inputs[profile_name + "_polar:CD"],
            )

            alpha_list[idx, :] = alpha_interp
            cl_list[idx, :] = np.interp(alpha_interp, alpha_element, cl_element)
            cl_inv_list[idx, :] = np.interp(alpha_interp, alpha_element, cl_inv_element)
            cd_list[idx, :] = np.interp(alpha_interp, alpha_element, cd_element)
            cd_min_list[idx] = inputs[profile_name + "_polar:CD_min_2D"]

        thrust, eta, torque = self.compute_pitch_performance_mod(
            inputs,
            theta_75,
            speed,
            altitude,
            omega,
            radius,
            alpha_list,
            cl_list,
            cd_list,
            cl_inv_list,
            cd_min_list,
        )
        power = torque * omega * np.pi / 30.0

        outputs["data:aerodynamics:propeller:point_performance:thrust"] = thrust
        outputs["data:aerodynamics:propeller:point_performance:efficiency"] = eta
        outputs["data:aerodynamics:propeller:point_performance:power"] = power

    def find_altitude(self, density):

        alt = fsolve(self.diff_to_density, x0=np.array([0.0]), args=density)
        return max(float(alt[0]), 0.0)

    @staticmethod
    def diff_to_density(altitude, target_density):

        return Atmosphere(altitude, altitude_in_feet=False).density - target_density

    def compute_pitch_performance_mod(
        self,
        inputs,
        theta_75,
        v_inf,
        altitude,
        omega,
        radius,
        alpha_list,
        cl_list,
        cd_list,
        cl_inv_list,
        cd_min_list,
    ):

        """
        This function calculates the thrust, efficiency and power at a given flight speed,
        altitude h and propeller angular speed.

        :param inputs: structure of data relative to the blade geometry available from setup
        :param theta_75: pitch defined at r = 0.75*R radial position [deg].
        :param v_inf: flight speeds [m/s].
        :param altitude: flight altitude [m].
        :param omega: angular velocity of the propeller [RPM].
        :param radius: array of radius of discretized blade elements [m].
        :param alpha_list: angle of attack list for aerodynamic coefficient of profile at
        discretized blade element [deg].
        :param cl_list: cl list for aerodynamic coefficient of profile at discretized blade
        element [-].
        :param cd_list: cd list for aerodynamic coefficient of profile at discretized blade
        element [-].
        :param cl_inv_list: cl list for inviscid aerodynamic coefficient of profile at discretized
        blade
        element [-].
        :param cd_min_list: cd_min list for aerodynamic coefficient of profile at discretized blade
        element [-].

        :return: thrust [N], eta (efficiency) [-] and power [W].
        """

        blades_number = inputs["data:geometry:propeller:blades_number"]
        radius_min = inputs["data:geometry:propeller:hub_diameter"] / 2.0
        radius_max = inputs["data:geometry:propeller:diameter"] / 2.0
        sweep_vect = inputs["data:geometry:propeller:sweep_vect"]
        chord_vect = inputs["data:geometry:propeller:chord_vect"]
        twist_vect = inputs["data:geometry:propeller:twist_vect"]
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        reference_reynolds = inputs["reference_reynolds"]
        length = radius_max - radius_min
        element_length = length / self.options["elements_number"]
        omega = omega * np.pi / 30.0
        atm = Atmosphere(altitude, altitude_in_feet=False)

        theta_75_ref = np.interp(0.75, radius_ratio_vect, twist_vect)

        # Initialise vectors
        vi_vect = np.zeros_like(radius)
        vt_vect = np.zeros_like(radius)
        thrust_element_vector = np.zeros_like(radius)
        torque_element_vector = np.zeros_like(radius)
        alpha_vect = np.zeros_like(radius)
        speed_vect = np.array([0.1 * float(v_inf), 1.0])

        chord = np.interp(radius / radius_max, radius_ratio_vect, chord_vect)

        theta = np.interp(radius / radius_max, radius_ratio_vect, twist_vect) + (
            theta_75 - theta_75_ref
        )
        sweep = np.interp(radius / radius_max, radius_ratio_vect, sweep_vect)

        # Loop on element number to compute equations
        for idx, _ in enumerate(radius):

            # Solve BEM vs. disk theory system of equations
            speed_vect = fsolve(
                self.delta_mod,
                speed_vect,
                (
                    radius[idx],
                    radius_min,
                    radius_max,
                    chord[idx],
                    blades_number,
                    sweep[idx],
                    omega,
                    v_inf,
                    theta[idx],
                    alpha_list[idx, :],
                    cl_list[idx, :],
                    cd_list[idx, :],
                    cl_inv_list[idx, :],
                    cd_min_list[idx],
                    atm,
                    reference_reynolds,
                ),
                xtol=1e-3,
            )
            vi_vect[idx] = speed_vect[0]
            vt_vect[idx] = speed_vect[1]

            results = self.bem_theory_mod(
                speed_vect,
                radius[idx],
                chord[idx],
                blades_number,
                sweep[idx],
                omega,
                v_inf,
                theta[idx],
                alpha_list[idx, :],
                cl_list[idx, :],
                cd_list[idx, :],
                cl_inv_list[idx, :],
                cd_min_list[idx],
                atm,
                reference_reynolds,
            )
            out_of_polars = results[3]

            if out_of_polars:
                thrust_element_vector[idx] = 0.0
                torque_element_vector[idx] = 0.0
                # print(out_of_polars)
            else:
                thrust_element_vector[idx] = results[0] * element_length * atm.density
                torque_element_vector[idx] = results[1] * element_length * atm.density
            alpha_vect[idx] = results[2]

        torque = np.sum(torque_element_vector)
        thrust = float(np.sum(thrust_element_vector))
        power = float(torque * omega)
        eta = float(v_inf * thrust / power)

        return thrust, eta, torque

    @staticmethod
    def bem_theory_mod(
        speed_vect: np.array,
        radius: float,
        chord: float,
        blades_number: float,
        sweep: float,
        omega: float,
        v_inf: float,
        theta: float,
        alpha_element: np.array,
        cl_element: np.array,
        cd_element: np.array,
        cl_inv_element: np.array,
        cd_min_element: float,
        atm: Atmosphere,
        reference_reynolds: float,
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element,
        its aerodynamic polars, flight conditions and axial/tangential velocities it computes the
        thrust and the torque produced using force and momentum with BEM theory.

        :param speed_vect: the vector of axial and tangential induced speed in m/s
        :param radius: radius position of the element center  [m]
        :param chord: chord at the center of element [m]
        :param blades_number: number of blades [-]
        :param sweep: sweep angle [deg.]
        :param omega: angular speed of propeller [rad/sec]
        :param v_inf: flight speed [m/s]
        :param theta: profile angle relative to aircraft airflow v_inf [deg.]
        :param alpha_element: reference angle vector for element polars [deg.]
        :param cl_element: cl vector for element [-]
        :param cd_element: cd vector for element [-]
        :param cl_inv_element: inviscid cl vector for element [-]
        :param cd_min_element: min 2d cd for element [-]
        :param atm: atmosphere properties
        :param reference_reynolds: Reynolds number at which the aerodynamic properties were computed

        :return: The calculated dT/(rho*dr) and dQ/(rho*dr) increments with BEM method.
        """

        # Extract axial/tangential speeds
        v_i = speed_vect[0]
        v_t = speed_vect[1]

        # Calculate speed composition and relative air angle (in deg.)
        v_ax = v_inf + v_i
        v_t = (omega * radius - v_t) * np.cos(sweep * np.pi / 180.0)
        rel_fluid_speed = np.sqrt(v_ax ** 2.0 + v_t ** 2.0)
        phi = np.arctan(v_ax / v_t)
        alpha = theta - phi * 180.0 / np.pi

        # Compute local mach
        atm.true_airspeed = rel_fluid_speed
        mach_local = atm.mach

        # Apply the compressibility corrections for cl and cd
        out_of_polars = bool((alpha > max(alpha_element)) or (alpha < min(alpha_element)))

        c_l = np.interp(alpha, alpha_element, cl_element)
        c_l_inv = np.interp(alpha, alpha_element, cl_inv_element)
        c_d = np.interp(alpha, alpha_element, cd_element)

        if mach_local < 1:
            beta = np.sqrt(1 - mach_local ** 2.0)
            c_l = c_l / (beta + c_l * mach_local ** 2.0 / (2.0 + 2.0 * beta))
            c_l_inv = c_l_inv / (beta + c_l_inv * mach_local ** 2.0 / (2.0 + 2.0 * beta))
            c_d = c_d / (beta + c_d * mach_local ** 2.0 / (2.0 + 2.0 * beta))
            c_d_min = cd_min_element / (
                beta + cd_min_element * mach_local ** 2.0 / (2.0 + 2.0 * beta)
            )
        else:
            beta = np.sqrt(mach_local ** 2.0 - 1)
            c_l = c_l / beta
            c_l_inv = c_l_inv / beta
            c_d = c_d / beta
            c_d_min = cd_min_element / beta

        reynolds = chord * atm.unitary_reynolds
        f_re = (3.46 * np.log(reynolds) - 5.6) ** -2
        f_re_t = (3.46 * np.log(reference_reynolds) - 5.6) ** -2
        c_d = c_d * (f_re / f_re_t)
        c_d_min = c_d_min * (f_re / f_re_t)

        delta_c_l = c_l_inv - c_l
        delta_c_d = c_d - c_d_min

        c_l_3d = c_l + 3.0 * (chord / radius) ** 2.0 * np.cos(theta) ** 4.0 * delta_c_l
        c_d_3d = c_d + 3.0 * (chord / radius) ** 2.0 * np.cos(theta) ** 4.0 * delta_c_d

        # Calculate force and momentum
        thrust_element = (
            0.5
            * blades_number
            * chord
            * rel_fluid_speed ** 2.0
            * (c_l_3d * np.cos(phi) - c_d_3d * np.sin(phi))
        )
        torque_element = (
            0.5
            * blades_number
            * chord
            * rel_fluid_speed ** 2.0
            * (c_l_3d * np.sin(phi) + c_d_3d * np.cos(phi))
            * radius
        )

        # Store results
        output = np.empty(4)
        output[0] = thrust_element
        output[1] = torque_element
        output[2] = alpha
        output[3] = out_of_polars

        return output

    # noinspection PyUnusedLocal
    @staticmethod
    def disk_theory_mod(
        speed_vect: np.array,
        radius: float,
        radius_min: float,
        radius_max: float,
        blades_number: float,
        sweep: float,
        omega: float,
        v_inf: float,
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element,
        its aerodynamic polars, flight conditions and axial/tangential velocities it computes the
        thrust and the torque produced using force and momentum with disk theory.

        :param speed_vect: the vector of axial and tangential induced speed in m/s
        :param radius: radius position of the element center  [m]
        :param radius_min: Hub radius [m]
        :param radius_max: Max radius [m]
        :param blades_number: number of blades [-]
        :param sweep: sweep angle [deg.]
        :param omega: angular speed of propeller [rad/sec]
        :param v_inf: flight speed [m/s]

        :return: The calculated dT/(rho*dr) and dQ/(rho*dr) increments with disk theory method.
        """

        # Extract axial/tangential speeds
        v_i = speed_vect[0]
        v_t = speed_vect[1]

        # Calculate speed composition and relative air angle (in deg.)
        v_ax = v_inf + v_i
        # Needed for the computation of the hub lost factor
        phi = np.arctan2(v_ax, max((omega * radius - v_t) * np.cos(sweep * np.pi / 180.0), 1e-6))

        # f_tip is the tip loss factor
        f_tip = (
            2
            / np.pi
            * np.arccos(
                np.exp(
                    -blades_number
                    / 2
                    * (
                        (radius_max - radius)
                        / radius
                        * np.sqrt(1 + (omega * radius / (v_ax + 1e-12 * (v_ax == 0.0))) ** 2.0)
                    )
                )
            )
        )

        # f_hub is the hub loss factor
        f_hub = np.clip(
            2
            / np.pi
            * np.arccos(
                np.exp(-blades_number / 2 * (radius - radius_min) / (radius * np.sin(phi)))
            ),
            0.0,
            1.0,
        )
        # f_hub = 1.0

        # Calculate force and momentum
        thrust_element = 4.0 * np.pi * radius * (v_inf + v_i) * v_i * f_tip * f_hub
        torque_element = 4.0 * np.pi * radius ** 2.0 * (v_inf + v_i) * v_t * f_tip * f_hub

        # Store results
        output = np.empty(2)
        output[0] = thrust_element
        output[1] = torque_element

        return output

    def delta_mod(
        self,
        speed_vect: np.array,
        radius: float,
        radius_min: float,
        radius_max: float,
        chord: float,
        blades_number: float,
        sweep: float,
        omega: float,
        v_inf: float,
        theta: float,
        alpha_element: np.array,
        cl_element: np.array,
        cd_element: np.array,
        cl_inv_element: np.array,
        cd_min_element: float,
        atm: Atmosphere,
        reference_reynolds: float,
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element,
        its aerodynamic polars, flight conditions and axial/tangential velocities it computes the
        thrust and the torque produced using force and momentum with disk theory.

        :param speed_vect: the vector of axial and tangential induced speed in m/s
        :param radius: radius position of the element center  [m]
        :param radius_min: Hub radius [m]
        :param radius_max: Max radius [m]
        :param chord: chord at the center of element [m]
        :param blades_number: number of blades [-]
        :param sweep: sweep angle [deg.]
        :param omega: angular speed of propeller [rad/sec]
        :param v_inf: flight speed [m/s]
        :param theta: profile angle relative to aircraft airflow v_inf [DEG]
        :param alpha_element: reference angle vector for element polars [DEG]
        :param cl_element: cl vector for element [-]
        :param cd_element: cd vector for element [-]
        :param cl_inv_element: cl list for inviscid aerodynamic coefficient of the profile
        element [-].
        :param cd_min_element: minimum drag coefficient of the profile [-].
        :param atm: atmosphere properties
        :param reference_reynolds: Reynolds number at which the aerodynamic properties were computed

        :return: The difference between BEM dual methods for dT/(rho*dr) and dQ/ increments.
        """

        bem_result = self.bem_theory_mod(
            speed_vect,
            radius,
            chord,
            blades_number,
            sweep,
            omega,
            v_inf,
            theta,
            alpha_element,
            cl_element,
            cd_element,
            cl_inv_element,
            cd_min_element,
            atm,
            reference_reynolds,
        )

        adt_result = self.disk_theory_mod(
            speed_vect, radius, radius_min, radius_max, blades_number, sweep, omega, v_inf
        )

        return bem_result[0:1] - adt_result

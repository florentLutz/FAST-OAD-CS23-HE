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
        self.add_subsystem("chord_law", _PrepareChordToDiameterLaw(), promotes=["*"])
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

        self.add_subsystem(
            "similarity_ratios",
            _ComputeSimilarityParameters(elements_number=self.options["elements_number"]),
            promotes=["*"],
        )


class _ComputeSimilarityParameters(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("elements_number", default=20, types=int)

    def setup(self):
        self.add_input("data:geometry:propeller:radius_ratio_vect", val=np.nan, shape_by_conn=True)
        self.add_input(
            "data:geometry:propeller:chord_vect",
            val=np.nan,
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:blades_number", val=np.nan)

        self.add_output(
            "data:geometry:propeller:solidity_ratio",
            val=np.nan,
        )
        self.add_output(
            "data:geometry:propeller:activity_factor",
            val=np.nan,
        )
        self.add_output(
            "data:geometry:propeller:aspect_ratio",
            val=np.nan,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_diameter = inputs["data:geometry:propeller:diameter"]
        radius_max = propeller_diameter / 2.0
        radius_min = 0.2 * radius_max
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        radius_vect = radius_ratio_vect * radius_max
        chord_vect = inputs["data:geometry:propeller:chord_vect"]
        n_blades = inputs["data:geometry:propeller:blades_number"]

        length = radius_max - radius_min
        elements_number = np.arange(self.options["elements_number"])
        element_length = length / self.options["elements_number"]
        radius = radius_min + (elements_number + 0.5) * element_length

        chord_array = np.interp(radius, radius_vect, chord_vect)

        solidity = n_blades / np.pi / radius_max ** 2.0 * np.sum(chord_array * element_length)
        activity_factor = (
            100000 / 32 / radius_max ** 5.0 * np.sum(chord_array * radius ** 3.0 * element_length)
        )
        c_star = np.sum(chord_array * radius ** 2.0 * element_length) / np.sum(
            radius ** 2.0 * element_length
        )
        aspect_ratio = radius_max / c_star

        outputs["data:geometry:propeller:solidity_ratio"] = solidity
        outputs["data:geometry:propeller:activity_factor"] = activity_factor
        outputs["data:geometry:propeller:aspect_ratio"] = aspect_ratio


class _PrepareChordToDiameterLaw(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:propeller:radius_ratio_vect", val=np.nan, shape_by_conn=True)
        self.add_input(
            "data:geometry:propeller:mid_chord_ratio",
            val=np.nan,
            desc="Ratio of the maximum chord to diameter to the chord to diameter of the root",
        )
        self.add_input(
            "data:geometry:propeller:mid_chord_radius_ratio",
            val=np.nan,
            desc="Position of the point of maximum chord as a ratio of the radius",
        )

        self.add_output(
            "data:geometry:propeller:chord_to_diameter_vect",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        chord_to_diameter_root = 0.0380
        root_radius_ratio = 0.2
        chord_to_diameter_tip = 0.0380
        tip_radius_ratio = 0.999

        radius_ratio_mid = float(inputs["data:geometry:propeller:mid_chord_radius_ratio"])
        chord_to_diameter_ratio_mid = (
            float(inputs["data:geometry:propeller:mid_chord_ratio"]) * chord_to_diameter_root
        )

        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]

        matrix_to_inv = np.array(
            [
                [root_radius_ratio ** 2.0, root_radius_ratio, 1.0, 0.0, 0.0, 0.0],
                [radius_ratio_mid ** 2.0, radius_ratio_mid, 1.0, 0.0, 0.0, 0.0],
                [2.0 * radius_ratio_mid, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0 * radius_ratio_mid, 1.0, 0.0],
                [0.0, 0.0, 0.0, radius_ratio_mid ** 2.0, radius_ratio_mid, 1.0],
                [0.0, 0.0, 0.0, tip_radius_ratio ** 2.0, tip_radius_ratio, 1.0],
            ]
        )
        result_matrix = np.array(
            [
                [chord_to_diameter_root],
                [chord_to_diameter_ratio_mid],
                [
                    (chord_to_diameter_tip - chord_to_diameter_root)
                    / (tip_radius_ratio - root_radius_ratio)
                ],
                [
                    (chord_to_diameter_tip - chord_to_diameter_root)
                    / (tip_radius_ratio - root_radius_ratio)
                ],
                [chord_to_diameter_ratio_mid],
                [chord_to_diameter_tip],
            ]
        )

        k12, k11, k10, k22, k21, k20 = np.dot(
            np.linalg.inv(matrix_to_inv), result_matrix
        ).transpose()[0]
        chord_distribution = np.where(
            radius_ratio_vect < radius_ratio_mid,
            k12 * radius_ratio_vect ** 2.0 + k11 * radius_ratio_vect + k10,
            k22 * radius_ratio_vect ** 2.0 + k21 * radius_ratio_vect + k20,
        )

        chord_distribution[0] = chord_distribution[1]

        outputs["data:geometry:propeller:chord_to_diameter_vect"] = chord_distribution


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

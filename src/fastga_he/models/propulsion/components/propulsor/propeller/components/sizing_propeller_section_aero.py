# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar, POLAR_POINT_COUNT


class SizingPropellerSectionAero(om.Group):
    def initialize(self):

        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )
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
        self.options.declare("elements_number", default=7, types=int)

    def setup(self):
        ivc = om.IndepVarComp()
        ivc.add_output("data:aerodynamics:propeller:mach", val=0.0)
        ivc.add_output("data:aerodynamics:propeller:reynolds", val=1e6)
        self.add_subsystem("propeller_aero_conditions", ivc, promotes=["*"])

        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar",
                XfoilPolar(
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.connect("data:aerodynamics:propeller:mach", profile + "_polar.xfoil:mach")
            self.connect("data:aerodynamics:propeller:reynolds", profile + "_polar.xfoil:reynolds")
            self.connect(
                profile + "_polar.xfoil:alpha", "propeller_aero." + profile + "_polar:alpha"
            )
            self.connect(profile + "_polar.xfoil:CL", "propeller_aero." + profile + "_polar:CL")
            self.connect(profile + "_polar.xfoil:CD", "propeller_aero." + profile + "_polar:CD")

        self.add_subsystem(
            "propeller_aero",
            SizingPropellerSectionAeroIdentification(
                propeller_id=self.options["propeller_id"],
                sections_profile_position_list=self.options["sections_profile_position_list"],
                sections_profile_name_list=self.options["sections_profile_name_list"],
                elements_number=self.options["elements_number"],
            ),
            promotes_inputs=["data:*"],
            promotes_outputs=["*"],
        )


class SizingPropellerSectionAeroIdentification(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )
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
        self.options.declare("elements_number", default=7, types=int)

    def setup(self):

        propeller_id = self.options["propeller_id"]
        elements_number = self.options["elements_number"]

        for profile in self.options["sections_profile_name_list"]:
            self.add_input(
                profile + "_polar:alpha", val=np.nan, units="deg", shape=POLAR_POINT_COUNT
            )
            self.add_input(profile + "_polar:CL", val=np.nan, shape=POLAR_POINT_COUNT)
            self.add_input(profile + "_polar:CD", val=np.nan, shape=POLAR_POINT_COUNT)

            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":alpha_list",
                wrt=profile + "_polar:alpha",
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_array",
                wrt=profile + "_polar:CL",
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:propeller:" + propeller_id + ":cd_array",
                wrt=profile + "_polar:CD",
            )

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            shape=elements_number,
            units="m",
            val=np.full(elements_number, np.nan),
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":alpha_list",
            val=np.ones(POLAR_POINT_COUNT),
            shape=POLAR_POINT_COUNT,
            units="deg",
        )
        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_array",
            val=np.ones((elements_number, POLAR_POINT_COUNT)),
            shape=(elements_number, POLAR_POINT_COUNT),
        )
        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":cd_array",
            val=np.ones((elements_number, POLAR_POINT_COUNT)),
            shape=(elements_number, POLAR_POINT_COUNT),
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]
        elements_number = self.options["elements_number"]

        radius_max = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"] / 2.0
        )
        elements_radius = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"
        ]

        alpha_interp = np.array([0])

        for profile in self.options["sections_profile_name_list"]:
            alpha_interp = np.union1d(alpha_interp, inputs[profile + "_polar:alpha"])

        alpha_list = np.zeros(POLAR_POINT_COUNT)
        cl_list = np.zeros((elements_number, POLAR_POINT_COUNT))
        cd_list = np.zeros((elements_number, POLAR_POINT_COUNT))

        valid_idx = len(alpha_interp)

        for idx, value in enumerate(elements_radius):

            index = np.where(
                self.options["sections_profile_position_list"] <= (value / radius_max)
            )[0]
            if index is None:
                profile_name = self.options["sections_profile_position_list"][0]
            else:
                profile_name = self.options["sections_profile_name_list"][int(index[-1])]

            # Load profile polars
            alpha_element, cl_element, cd_element = self.reshape_polar(
                inputs[profile_name + "_polar:alpha"],
                inputs[profile_name + "_polar:CL"],
                inputs[profile_name + "_polar:CD"],
            )

            cl_list[idx, :valid_idx] = np.interp(alpha_interp, alpha_element, cl_element)
            cd_list[idx, :valid_idx] = np.interp(alpha_interp, alpha_element, cd_element)

        alpha_list[:valid_idx] = alpha_interp
        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":alpha_list"
        ] = alpha_list
        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_array"] = cl_list
        outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":cd_array"] = cd_list

    @staticmethod
    def reshape_polar(alpha, c_l, c_d):
        """
        Reads the polar under the openmdao format (meaning with additional zeros and reshape
        so that only relevant angle are considered.

        Assumes that the AOA list is ordered.
        """
        idx_start = np.argmin(alpha)
        idx_end = np.argmax(alpha)

        return (
            alpha[idx_start : idx_end + 1],
            c_l[idx_start : idx_end + 1],
            c_d[idx_start : idx_end + 1],
        )

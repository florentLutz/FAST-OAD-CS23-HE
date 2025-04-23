# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingHighRPMICEDrag(om.ExplicitComponent):
    """
    Additional drag coefficient due to the installation of the ICE depending on the location,
    inside the nose, it will be computed as not contributing. Based on the formula from
    :cite:`gudmundsson:2013` for the form drag and :cite:`roskampart6:1985` for the interference
    drag.
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the generator, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input("data:aerodynamics:" + ls_tag + ":mach", val=np.nan)
        self.add_input("data:aerodynamics:" + ls_tag + ":unit_reynolds", val=np.nan, units="m**-1")

        if self.options["position"] == "on_the_wing":
            self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

            self.add_input(
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:length",
                val=np.nan,
                desc="Length of the ICE nacelle",
                units="m",
            )
            self.add_input(
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width",
                val=np.nan,
                desc="Width of the ICE nacelle",
                units="m",
            )
            self.add_input(
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:height",
                val=np.nan,
                desc="Height of the ICE nacelle",
                units="m",
            )
            self.add_input(
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:wet_area",
                val=np.nan,
                desc="Height of the ICE nacelle",
                units="m**2",
            )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":"
            + ls_tag
            + ":CD0",
            val=0.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if self.options["position"] == "on_the_wing":
            nacelle_length = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length"
            ]
            nacelle_height = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:height"
            ]
            nacelle_width = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width"
            ]
            nacelle_wet_area = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:wet_area"
            ]

            mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]
            reynolds = inputs["data:aerodynamics:" + ls_tag + ":unit_reynolds"] * nacelle_length

            wing_area = inputs["data:geometry:wing:area"]
            l0_wing = inputs["data:geometry:wing:MAC:length"]

            cf_nac = 0.455 / ((1 + 0.144 * mach**2) ** 0.65 * (np.log10(reynolds)) ** 2.58)

            fineness_ratio = nacelle_length / np.sqrt(4 * nacelle_height * nacelle_width / np.pi)

            form_factor = 1 + 0.35 / fineness_ratio
            interference_factor = 1.2

            cd0_form = cf_nac * form_factor * nacelle_wet_area * interference_factor
            cd0_interference = 0.036 * l0_wing * nacelle_width * 0.2**2.0

            cd0 = (cd0_interference + cd0_form) / wing_area

            outputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0"
            ] = cd0

        else:
            outputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0"
            ] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if self.options["position"] == "on_the_wing":
            nacelle_length = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length"
            ]
            nacelle_height = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:height"
            ]
            nacelle_width = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width"
            ]
            nacelle_wet_area = inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:wet_area"
            ]

            mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]
            unit_reynolds = inputs["data:aerodynamics:" + ls_tag + ":unit_reynolds"]

            l0_wing = inputs["data:geometry:wing:MAC:length"]
            wing_area = inputs["data:geometry:wing:area"]

            reynolds = unit_reynolds * nacelle_length

            d_reynolds_d_unit_reynolds = nacelle_length
            d_reynolds_d_length = unit_reynolds

            cf_nac = 0.455 / ((1 + 0.144 * mach**2) ** 0.65 * (np.log10(reynolds)) ** 2.58)

            d_cf_d_mach = -(
                0.288
                * 0.455
                * 0.65
                * (np.log10(reynolds)) ** -2.58
                * (1 + 0.144 * mach**2) ** -1.65
                * mach
            )
            d_cf_d_reynolds = (
                (-2.58 * 0.455 * (1 + 0.144 * mach**2) ** -0.65 * (np.log10(reynolds)) ** -3.58)
                / reynolds
                / np.log(10)
            )

            d_fineness_d_length = 1.0 / np.sqrt(4 * nacelle_height * nacelle_width / np.pi)
            d_fineness_d_height = (
                -0.5
                * nacelle_length
                / np.sqrt(4 * nacelle_width / np.pi)
                * nacelle_height ** (-3.0 / 2.0)
            )
            d_fineness_d_width = (
                -0.5
                * nacelle_length
                / np.sqrt(4 * nacelle_height / np.pi)
                * nacelle_width ** (-3.0 / 2.0)
            )

            fineness_ratio = nacelle_length / np.sqrt(4 * nacelle_height * nacelle_width / np.pi)

            d_form_d_fineness = -0.35 / fineness_ratio**2.0

            d_cd_interference_d_width = 0.036 * l0_wing * 0.2**2.0

            form_factor = 1 + 0.35 / fineness_ratio
            interference_factor = 1.2

            cd0_form = cf_nac * form_factor * nacelle_wet_area * interference_factor
            cd0_interference = 0.036 * l0_wing * nacelle_width * 0.2**2.0

            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:aerodynamics:" + ls_tag + ":mach",
            ] = (form_factor * nacelle_wet_area * interference_factor * d_cf_d_mach) / wing_area
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:aerodynamics:" + ls_tag + ":unit_reynolds",
            ] = (
                form_factor
                * nacelle_wet_area
                * interference_factor
                * d_cf_d_reynolds
                * d_reynolds_d_unit_reynolds
            ) / wing_area
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:length",
            ] = (
                nacelle_wet_area
                * interference_factor
                * (
                    form_factor * d_cf_d_reynolds * d_reynolds_d_length
                    + cf_nac * d_form_d_fineness * d_fineness_d_length
                )
            ) / wing_area
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:height",
            ] = (
                nacelle_wet_area
                * interference_factor
                * cf_nac
                * d_form_d_fineness
                * d_fineness_d_height
                / wing_area
            )
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width",
            ] = (
                nacelle_wet_area
                * interference_factor
                * cf_nac
                * d_form_d_fineness
                * d_fineness_d_width
                + d_cd_interference_d_width
            ) / wing_area
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":nacelle:wet_area",
            ] = cf_nac * form_factor * interference_factor / wing_area
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:geometry:wing:area",
            ] = -(cd0_interference + cd0_form) / wing_area**2.0
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:geometry:wing:MAC:length",
            ] = 0.036 * nacelle_width * 0.2**2.0 / wing_area

        else:
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:aerodynamics:" + ls_tag + ":mach",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":"
                + ls_tag
                + ":CD0",
                "data:aerodynamics:" + ls_tag + ":unit_reynolds",
            ] = 0.0

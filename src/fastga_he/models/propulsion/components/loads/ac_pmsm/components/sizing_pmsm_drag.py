# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import logging

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION

_LOGGER = logging.getLogger(__name__)


class SizingPMSMDrag(om.ExplicitComponent):
    """
     Additional drag coefficient due to the installation of the PMSM depending on the location,
    inside the nose, it will be computed as not contributing, just like we did it for ICE,
    turboprop, ... If it is on the wing, is will be computed considering it has a fairing going
    beyond the length to avoid having a "pancake" on the wing. Based on the formula from
    :cite:`gudmundsson:2013`.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the pmsm, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        # Not as useful as the ones in aerodynamics, here it will just be run twice in the sizing
        # group
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        if position == "on_the_wing":
            self.add_input(
                name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
                val=np.nan,
                units="m",
            )
            self.add_input(
                name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":fairing:fineness",
                val=1.75,
                desc="Ratio between the fairing length and the motor diameter",
            )
            self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
            self.add_input("data:aerodynamics:" + ls_tag + ":mach", val=np.nan)
            self.add_input(
                "data:aerodynamics:" + ls_tag + ":unit_reynolds", val=np.nan, units="m**-1"
            )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )

    def setup_partials(self):
        if self.options["position"] == "on_the_wing":
            pmsm_id = self.options["pmsm_id"]
            ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

            self.declare_partials(
                of="*",
                wrt=[
                    "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
                    "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":fairing:fineness",
                    "data:geometry:wing:area",
                    "data:aerodynamics:" + ls_tag + ":mach",
                    "data:aerodynamics:" + ls_tag + ":unit_reynolds",
                ],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        motor_diameter = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]

        if position == "on_the_wing":
            motor_length = inputs[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"
            ]
            fineness = inputs[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":fairing:fineness"
            ]

            if motor_length > fineness * motor_diameter:
                _LOGGER.warning(
                    "Inputs does not allow for a fairing long enough to house the motor, consider "
                    "changing the fineness"
                )

            wing_area = inputs["data:geometry:wing:area"]
            mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]
            reynolds = (
                inputs["data:aerodynamics:" + ls_tag + ":unit_reynolds"] * fineness * motor_diameter
            )

            # Complete turbulent flow with compressibility correction, from Gudmundsson page 678
            cf = 0.455 / (np.log10(reynolds) ** 2.58 * (1.0 + 0.144 * mach**2.0) ** 0.65)
            interference_factor = 1.1

            # From Gudmundsson page 703
            form_factor = 1.0 + 0.35 / fineness

            fairing_wet_area = np.pi * motor_diameter**2.0 * (1.0 / 4.0 + fineness)
            cd0 = interference_factor * cf * form_factor * fairing_wet_area / wing_area

            # TODO: This coefficient is for the nacelle alone, to account for installation on the
            #  wing it should take into account a delta (See Roskam part VI section 4.5.2.1),
            #  which depends on the installation angle which we do not yet consider.

            outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0"] = (
                cd0
            )

        else:
            outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0"] = (
                0.0
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        motor_diameter = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]

        if position == "on_the_wing":
            fineness = inputs[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":fairing:fineness"
            ]
            wing_area = inputs["data:geometry:wing:area"]
            mach = inputs["data:aerodynamics:" + ls_tag + ":mach"]
            unit_reynolds = inputs["data:aerodynamics:" + ls_tag + ":unit_reynolds"]
            reynolds = unit_reynolds * fineness * motor_diameter

            # Complete turbulent flow with compressibility correction, from Gudmundsson page 678
            cf = 0.455 / (np.log10(reynolds) ** 2.58 * (1.0 + 0.144 * mach**2.0) ** 0.65)
            d_cf_d_reynolds = (
                -2.58
                * 0.455
                / (1.0 + 0.144 * mach**2.0) ** 0.65
                * np.log10(reynolds) ** -3.58
                / (np.log(10) * reynolds)
            )
            interference_factor = 1.1

            # From Gudmundsson page 703
            form_factor = 1.0 + 0.35 / fineness

            fairing_wet_area = np.pi * motor_diameter**2.0 * (1.0 / 4.0 + fineness)

            partials[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            ] = (
                interference_factor
                * form_factor
                / wing_area
                * (
                    unit_reynolds * fineness * d_cf_d_reynolds * fairing_wet_area
                    + 2.0 * cf * np.pi * motor_diameter * (1.0 / 4.0 + fineness)
                )
            )

            partials[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":fairing:fineness",
            ] = (
                interference_factor
                / wing_area
                * (
                    unit_reynolds
                    * motor_diameter
                    * d_cf_d_reynolds
                    * form_factor
                    * fairing_wet_area
                    - 0.35 * cf / fineness**2.0 * fairing_wet_area
                    + cf * form_factor * np.pi * motor_diameter**2.0
                )
            )

            partials[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0",
                "data:geometry:wing:area",
            ] = -interference_factor * cf * form_factor * fairing_wet_area / wing_area**2.0

            partials[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0",
                "data:aerodynamics:" + ls_tag + ":mach",
            ] = (
                -0.65
                * interference_factor
                * form_factor
                * fairing_wet_area
                / wing_area
                * 0.455
                / np.log10(reynolds) ** 2.58
                * (1.0 + 0.144 * mach**2.0) ** -1.65
                * 0.144
                * 2.0
                * mach
            )

            partials[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":" + ls_tag + ":CD0",
                "data:aerodynamics:" + ls_tag + ":unit_reynolds",
            ] = (
                interference_factor
                * d_cf_d_reynolds
                * form_factor
                * fairing_wet_area
                / wing_area
                * motor_diameter
                * fineness
            )

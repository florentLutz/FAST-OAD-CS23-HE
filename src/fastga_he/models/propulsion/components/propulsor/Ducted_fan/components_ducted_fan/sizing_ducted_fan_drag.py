# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
sizing_ducted_fan_drag.py
===========================
METHOD -- classic component buildup (Raymer / Torenbeek nacelle drag), the standard conceptual
design approach for a body of revolution like a duct/nacelle:

    CD0 = Cf * FF * IF * (Swet / Sref)

    Cf   : turbulent flat-plate skin friction coefficient, from a Reynolds number based on the
           duct's axial length ("depth", already used by sizing_ducted_fan_cg_x.py):
               Re = reference_velocity * depth / nu(reference_altitude)
               Cf = 0.455 / log10(Re)**2.58                          (Prandtl-Schlichting)
           nu comes from stdatm.AtmosphereWithPartials.kinematic_viscosity.
    FF   : nacelle form factor (Raymer), function of the fineness ratio depth/diameter:
               FF = 1 + 0.35 / (depth / diameter)
    IF   : installation interference factor (tunable "settings:" input, default 1.25 -- typical
           for a wing-mounted nacelle/pod; put here rather than hardcoded so it can be tuned/
           calibrated later).
    Swet : external wetted area of the duct, approximated as a cylinder: pi * diameter * depth.
    Sref : wing reference area (data:geometry:wing:area), to match the propeller's CD0
           convention (an aircraft-level drag coefficient increment, not a fan-local one).

REFERENCE VELOCITY/ALTITUDE ARE SCALARS, NOT THE MISSION PROFILE -- this component lives in the
SIZING group (like ..components.sizing_propeller_drag.SizingPropellerDrag), which runs ONCE, not
per mission point -- it has no access to the vectorized true_airspeed/altitude arrays used in the
PerformancesDuctedFan group. "reference_velocity"/"reference_altitude" are single representative
conditions per regime (settings: inputs, one pair for low_speed, one for cruise), not a per-point
recompute. This is an accepted simplification for a CD0 buildup at this level of fidelity: Cf only
depends on Re through log10(Re)**-2.58, a very flat function, so the exact speed within a regime
(e.g. 60 vs 75 m/s in cruise) barely changes Cf. The resulting CD0 is meant to feed a low_speed
polar and a cruise polar downstream, each evaluated at its own representative condition, same as
elsewhere in this framework's sizing-time aero components.

"""

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials

from .constants import POSSIBLE_POSITION


class SizingDuctedFanDrag(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the ducted fan, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        # Kept for interface symmetry with SizingDuctedFanCGX/CGY/SizingPropellerDrag even
        # though the drag buildup below does not depend on position.
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the ducted fan",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":depth",
            val=np.nan,
            units="m",
            desc="Axial depth (length) of the duct",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":interference_factor",
            val=1.25,
            desc="Installation interference factor of the duct on the wing/airframe",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":"
            + ls_tag
            + ":reference_velocity",
            val=np.nan,
            units="m/s",
            desc="Reference airspeed used to build the duct's external Reynolds number for the "
            + ls_tag
            + " condition",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":"
            + ls_tag
            + ":reference_altitude",
            val=0.0,
            units="m",
            desc="Reference altitude used to build the duct's external Reynolds number for the "
            + ls_tag
            + " condition",
        )
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":" + ls_tag + ":CD0",
            val=0.001,
            desc="Parasite drag coefficient increment (referenced to wing area) of the duct's "
            "external surface, " + ls_tag + " condition",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def _buildup(self, inputs):
        ducted_fan_id = self.options["ducted_fan_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ][0]
        depth = inputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":depth"][0]
        interference_factor = inputs[
            "settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":interference_factor"
        ][0]
        velocity = inputs[
            "settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":"
            + ls_tag
            + ":reference_velocity"
        ][0]
        altitude = inputs[
            "settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":"
            + ls_tag
            + ":reference_altitude"
        ][0]
        sref = inputs["data:geometry:wing:area"][0]

        nu = AtmosphereWithPartials(altitude, altitude_in_feet=False).kinematic_viscosity

        reynolds = velocity * depth / nu
        log_re = np.log10(reynolds)
        cf = 0.455 / log_re**2.58

        fineness = depth / diameter
        form_factor = 1.0 + 0.35 / fineness

        swet = np.pi * diameter * depth

        cd0 = cf * form_factor * interference_factor * swet / sref

        return dict(
            diameter=diameter,
            depth=depth,
            interference_factor=interference_factor,
            velocity=velocity,
            altitude=altitude,
            sref=sref,
            nu=nu,
            reynolds=reynolds,
            log_re=log_re,
            cf=cf,
            form_factor=form_factor,
            swet=swet,
            cd0=cd0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        s = self._buildup(inputs)

        outputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":" + ls_tag + ":CD0"
        ] = s["cd0"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"
        s = self._buildup(inputs)

        diameter = s["diameter"]
        depth = s["depth"]
        interference_factor = s["interference_factor"]
        velocity = s["velocity"]
        nu = s["nu"]
        reynolds = s["reynolds"]
        log_re = s["log_re"]
        cf = s["cf"]
        form_factor = s["form_factor"]
        swet = s["swet"]
        cd0 = s["cd0"]
        sref = s["sref"]

        atm = AtmosphereWithPartials(s["altitude"], altitude_in_feet=False)
        d_nu_d_alt = atm.partial_kinematic_viscosity_altitude

        # d(Cf)/d(Re) -- Cf = 0.455 * log_re**-2.58
        d_cf_d_re = -2.58 * 0.455 * log_re ** (-3.58) / (reynolds * np.log(10.0))

        d_re_d_v = depth / nu
        d_re_d_depth = velocity / nu
        d_re_d_alt = -reynolds / nu * d_nu_d_alt

        d_cf_d_v = d_cf_d_re * d_re_d_v
        d_cf_d_depth = d_cf_d_re * d_re_d_depth
        d_cf_d_alt = d_cf_d_re * d_re_d_alt

        # FF = 1 + 0.35 * diameter / depth
        d_ff_d_depth = -0.35 * diameter / depth**2.0
        d_ff_d_diameter = 0.35 / depth

        # Swet = pi * diameter * depth
        d_swet_d_diameter = np.pi * depth
        d_swet_d_depth = np.pi * diameter

        common = interference_factor / sref

        d_cd0_d_depth = common * (
            d_cf_d_depth * form_factor * swet
            + cf * d_ff_d_depth * swet
            + cf * form_factor * d_swet_d_depth
        )
        d_cd0_d_diameter = common * (
            cf * d_ff_d_diameter * swet + cf * form_factor * d_swet_d_diameter
        )
        d_cd0_d_v = common * d_cf_d_v * form_factor * swet
        d_cd0_d_alt = common * d_cf_d_alt * form_factor * swet
        d_cd0_d_if = cf * form_factor * swet / sref
        d_cd0_d_sref = -cd0 / sref

        cd0_name = (
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":" + ls_tag + ":CD0"
        )

        partials[
            cd0_name,
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
        ] = d_cd0_d_diameter
        partials[
            cd0_name, "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":depth"
        ] = d_cd0_d_depth
        partials[
            cd0_name,
            "settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":interference_factor",
        ] = d_cd0_d_if
        partials[
            cd0_name,
            "settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":"
            + ls_tag
            + ":reference_velocity",
        ] = d_cd0_d_v
        partials[
            cd0_name,
            "settings:propulsion:he_power_train:ducted_fan:"
            + ducted_fan_id
            + ":"
            + ls_tag
            + ":reference_altitude",
        ] = d_cd0_d_alt
        partials[cd0_name, "data:geometry:wing:area"] = d_cd0_d_sref

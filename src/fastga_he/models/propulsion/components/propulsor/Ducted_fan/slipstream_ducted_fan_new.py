# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class _SlipstreamDuctedFanDeltas(om.ExplicitComponent):
    """
    A deliberately simple slipstream (blown wing) model for the ducted fan, meant to be replaced
    later by a proper OpenVSP-based surrogate. It exists so the main expected benefit of a
    wing-blown DEP-style ducted fan installation (extra lift from the blown wing area) is not
    simply absent from the model in the meantime.

    Physics, in order:
      1. thrust_loading Tc = T / (rho * V0^2 * D^2) -- same formula as the propeller's
         slipstream_thrust_loading.py.
      2. axial induction factor at the fan disk, actuator disk momentum theory:
         a_p = 0.5 * (sqrt(1 + 8/pi * Tc) - 1) -- same formula as the propeller's
         slipstream_axial_induction_factor.py.
      3. lift_increase_ratio ~= a_p. This is the exact reduction of the propeller's full de Vries
         formula (slipstream_lift_increase_ratio.py) when the installation angle i_p = 0 (fan
         axis aligned with the freestream, reasonable for a wing-mounted axial fan) and the
         height impact factor beta = 1 (full immersion, no vertical offset between fan axis and
         wing chord): (1 - 0) * sqrt(1 + 2*a_p*cos(alpha) + a_p^2) - 1 -> for small alpha,
         cos(alpha) ~= 1, so this is sqrt((1+a_p)^2) - 1 = a_p. It also deliberately skips the
         propeller's downstream contraction-ratio propagation (SlipstreamPropellerContractionRatio
         & co) -- that model corrects for the streamtube evolving between the disk and a wing
         located some distance behind it, which matters for a nose propeller but much less for a
         ducted fan sitting right on/at the wing (small "from_LE").
      4. unblown_section_lift = cl_wing_clean (= CL0_wing + CL_alpha_wing * alpha, already
         computed once upstream by SlipstreamAirframeLiftClean in delta_from_pt_file.py and
         reused here, same as the propeller does via its own unblown_section_lift input in
         slipstream_section_lift.py). A cruder stand-in for the propeller's version, which further
         rescales this by a LOCAL (spanwise-station-specific) cl_clean_ref/CL_ref ratio built from
         an OpenVSP-derived spanwise CL distribution, and adds flap increments. Using the wing's
         overall (not spanwise-local) lift coefficient ignores that spanwise variation and flap
         effects -- an accepted simplification for this first pass.
      5. blown_area_ratio = D * wing_chord_ref / S_wing (contraction_ratio implicitly = 1, same
         simplification as step 3 -- no downstream streamtube contraction accounted for).
      6. delta_Cl = unblown_section_lift * lift_increase_ratio * blown_area_ratio.

    delta_Cd and delta_Cm are left at 0.0 -- this first pass only targets the main expected
    benefit (extra lift), not drag/moment, which would need their own (more speculative)
    approximations.
    """

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Declared but not used to change behavior (this simple model doesn't yet distinguish
        # low-speed vs cruise -- cl_wing_clean, computed upstream in delta_from_pt_file.py, is
        # already the correct low-speed-or-cruise value for whichever instance this is, so no
        # ls_tag switching is needed HERE anymore, unlike the first version of this file).
        # Kept for interface compatibility: the generic slipstream assembler
        # (delta_from_pt_file.py) unconditionally sets this option on every propulsor-class
        # component's Slipstream group.
        self.options.declare(
            "low_speed_aero",
            default=False,
            desc="Boolean to consider low speed aerodynamics",
            types=bool,
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        self.add_input("thrust", units="N", val=np.nan, shape=n)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=n)
        self.add_input("density", units="kg/m**3", val=np.nan, shape=n)
        self.add_input(
            "cl_wing_clean",
            val=np.nan,
            shape=n,
            desc="Unblown wing lift coefficient (CL0 + CL_alpha*alpha), computed once upstream "
            "in delta_from_pt_file.py and shared by every propulsor's slipstream component",
        )
        self.add_input(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
            val=np.nan,
            units="m",
        )
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("delta_Cl", val=np.zeros(n))
        self.add_output("delta_Cd", val=np.zeros(n))
        self.add_output("delta_Cm", val=np.zeros(n))

        self.declare_partials(
            of="delta_Cl",
            wrt=["thrust", "true_airspeed", "density", "cl_wing_clean"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="delta_Cl",
            wrt=[
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
                "data:geometry:wing:area",
            ],
            method="exact",
            rows=np.arange(n),
            cols=np.zeros(n),
        )
        # delta_Cd/delta_Cm are constant zero: no dependency on any input, so their (correct,
        # all-zero) partials are simply left undeclared rather than requiring a compute_partials
        # entry that would always be zero.

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ]
        chord_ref = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref"
        ]
        wing_area = inputs["data:geometry:wing:area"]
        unblown_section_lift = inputs["cl_wing_clean"]

        thrust = np.maximum(inputs["thrust"], np.ones(n))
        tas = np.maximum(inputs["true_airspeed"], np.full(n, 1e-2))
        rho = inputs["density"]

        thrust_loading = thrust / (rho * tas**2.0 * diameter**2.0)
        a_p = 0.5 * (np.sqrt(1.0 + 8.0 / np.pi * thrust_loading) - 1.0)

        blown_area_ratio = diameter * chord_ref / wing_area

        outputs["delta_Cl"] = unblown_section_lift * a_p * blown_area_ratio
        outputs["delta_Cd"] = np.zeros(n)
        outputs["delta_Cm"] = np.zeros(n)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ]
        chord_ref = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref"
        ]
        wing_area = inputs["data:geometry:wing:area"]
        unblown_section_lift = inputs["cl_wing_clean"]

        thrust = np.maximum(inputs["thrust"], np.ones(n))
        tas = np.maximum(inputs["true_airspeed"], np.full(n, 1e-2))
        rho = inputs["density"]

        thrust_loading = thrust / (rho * tas**2.0 * diameter**2.0)
        sqrt_term = np.sqrt(1.0 + 8.0 / np.pi * thrust_loading)
        a_p = 0.5 * (sqrt_term - 1.0)

        # d(a_p)/d(thrust_loading) -- same as the propeller's
        # SlipstreamPropellerAxialInductionFactor.compute_partials(): (2/pi)/sqrt(1+8/pi*Tc)
        d_ap_d_tc = (2.0 / np.pi) / sqrt_term

        d_tc_d_thrust = 1.0 / (rho * tas**2.0 * diameter**2.0)
        d_tc_d_tas = -2.0 * thrust / (rho * tas**3.0 * diameter**2.0)
        d_tc_d_rho = -thrust / (rho**2.0 * tas**2.0 * diameter**2.0)
        d_tc_d_d = -2.0 * thrust / (rho * tas**2.0 * diameter**3.0)

        blown_area_ratio = diameter * chord_ref / wing_area

        common = unblown_section_lift * blown_area_ratio  # multiplies d(a_p)/d(x) terms

        partials["delta_Cl", "thrust"] = common * d_ap_d_tc * d_tc_d_thrust
        partials["delta_Cl", "true_airspeed"] = common * d_ap_d_tc * d_tc_d_tas
        partials["delta_Cl", "density"] = common * d_ap_d_tc * d_tc_d_rho
        partials["delta_Cl", "cl_wing_clean"] = a_p * blown_area_ratio

        d_blown_area_ratio_d_d = chord_ref / wing_area
        d_blown_area_ratio_d_chord = diameter / wing_area
        d_blown_area_ratio_d_wing_area = -diameter * chord_ref / wing_area**2.0

        partials[
            "delta_Cl", "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ] = unblown_section_lift * (
            blown_area_ratio * d_ap_d_tc * d_tc_d_d + a_p * d_blown_area_ratio_d_d
        )
        partials[
            "delta_Cl",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":wing_chord_ref",
        ] = unblown_section_lift * a_p * d_blown_area_ratio_d_chord
        partials["delta_Cl", "data:geometry:wing:area"] = (
            unblown_section_lift * a_p * d_blown_area_ratio_d_wing_area
        )


class SlipstreamDuctedFan(om.Group):

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # "low_speed_aero" is set unconditionally on every propulsor-class component's Slipstream
        # group by the generic slipstream assembler (delta_from_pt_file.py), unlike
        # "flaps_position" which is gated by the SFR registry flag -- declared here for interface
        # compatibility with that assembler.
        self.options.declare(
            "low_speed_aero",
            default=False,
            desc="Boolean to consider low speed aerodynamics",
            types=bool,
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        number_of_points = self.options["number_of_points"]
        low_speed_aero = self.options["low_speed_aero"]

        self.add_subsystem(
            name="deltas",
            subsys=_SlipstreamDuctedFanDeltas(
                ducted_fan_id=ducted_fan_id,
                number_of_points=number_of_points,
                low_speed_aero=low_speed_aero,
            ),
            # "thrust" must be promoted (not just delta_Cl/delta_Cd/delta_Cm) -- the assembler's
            # self.connect("thrust_splitter." + name + "_thrust", name + ".thrust") resolves
            # "<name>.thrust" against THIS group's own promotion boundary, so "thrust" has to be
            # visible as "ducted_fan_1.thrust" (etc.) from the outside, not stay buried as
            # "deltas.thrust".
            promotes=["*"],
        )

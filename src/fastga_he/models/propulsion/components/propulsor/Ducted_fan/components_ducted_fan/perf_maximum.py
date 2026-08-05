# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
perf_maximum.py
================

SIMPLIFICATION vs the propeller's PerformancesMaximum: the propeller
model has dedicated upstream components (perf_advance_ratio.py,
perf_tip_mach.py, perf_torque.py) that already expose "advance_ratio",
"tip_mach" and "torque_in" as promoted mission variables, so its
PerformancesMaximum only has to take the max of each. perf_ducted_fan_new.py
does not have separate advance-ratio/tip-Mach/torque components (only
PerformancesRPM, PerformancesPowerCoefficient, PerformancesShaftPower,
PerformancesMaximum -- see that group file), so this component
recomputes the tip Mach number and advance ratio itself, from the
converged rpm and flight condition, using the SAME oblique-inflow
formulas as perf_rpm.py (unclipped this time -- the J_MIN/J_MAX/0.80
clipping in perf_rpm.py / perf_power_coefficient.py exists only to keep
the surrogate query inside its training domain, not to report true
extreme values here).

torque_max is a direct max of "torque_in", already a promoted mission variable computed
upstream by perf_torque.py (torque_in = shaft_power_in / (2*pi*rpm/60)) -- unlike
tip_mach/advance_ratio, it doesn't need recomputing here.
"""

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum tip Mach, advance ratio and rpm of the ducted fan over the
    mission.
    """

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the ducted fan",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
            val=0.4,
            desc="Duct rectification factor of the oblique inflow correction",
        )

        self.add_input("rpm", units="min**-1", val=np.full(n, np.nan))
        self.add_input("true_airspeed", units="m/s", val=np.full(n, np.nan))
        self.add_input("altitude", units="m", val=np.full(n, np.nan))
        self.add_input("alpha", units="rad", val=np.zeros(n), desc="Local flow incidence angle")
        self.add_input("torque_in", units="N*m", val=np.full(n, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":tip_mach_max",
            val=0.8,
            desc="Maximum value of the ducted fan tip mach number",
        )
        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":advance_ratio_max",
            val=1.0,
            desc="Maximum value of the ducted fan advance ratio",
        )
        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":rpm_max",
            units="min**-1",
            val=12000.0,
            desc="Maximum value of the ducted fan rpm",
        )
        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":torque_max",
            units="N*m",
            val=50.0,
            desc="Maximum value of the ducted fan torque",
        )
        # Per-point (unclipped) advance ratio, exposed so mission points sitting outside
        # perf_rpm.py's J_MIN/J_MAX=[0.01, 1.20] surrogate training range can be spotted -- those
        # get silently clipped before being fed to the surrogate, so this is the only place the
        # true (pre-clip) value is visible. Same formula as _tip_mach_and_advance_ratio() below;
        # exposed as a full per-point vector alongside its max (advance_ratio_max).
        self.add_output(
            "advance_ratio",
            val=np.full(n, 0.5),
            shape=n,
            desc="Ducted fan advance ratio (unclipped) at each mission point -- compare against "
            "perf_rpm.py's J_MIN=0.01/J_MAX=1.20 to spot points outside the surrogate's range",
        )

        # The 4 scalar *_max outputs are named explicitly (not of="*") since "advance_ratio"
        # below is a per-point, non-reduced output and needs a diagonal sparsity pattern, not
        # this "row 0 wrt every column" one.
        _max_outputs = [
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":tip_mach_max",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":advance_ratio_max",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":rpm_max",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":torque_max",
        ]
        self.declare_partials(
            of=_max_outputs,
            wrt=["rpm", "true_airspeed", "altitude", "alpha", "torque_in"],
            method="exact",
            rows=np.zeros(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of=_max_outputs,
            wrt=[
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
                "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
            ],
            method="exact",
        )
        # "advance_ratio" is a per-point (not reduced-to-scalar) output, so its sparsity pattern
        # is elementwise/diagonal against the vector inputs, not "row 0 wrt every column" like the
        # *_max outputs above -- needs its own declare_partials calls.
        self.declare_partials(
            of="advance_ratio",
            wrt=["rpm", "true_airspeed", "alpha"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="advance_ratio",
            wrt=[
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
                "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
            ],
            method="exact",
            rows=np.arange(n),
            cols=np.zeros(n),
        )
        # advance_ratio does not depend on altitude or torque_in -- no partials declared for
        # those (nothing to set in compute_partials() for this output/those inputs).

    def _tip_mach_and_advance_ratio(self, inputs):
        ducted_fan_id = self.options["ducted_fan_id"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ][0]
        k_duct = inputs[
            "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct"
        ][0]
        rpm = inputs["rpm"]
        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]
        alpha = inputs["alpha"]

        rps = rpm / 60.0
        factor = np.sqrt(np.cos(alpha) ** 2.0 + k_duct**2.0 * np.sin(alpha) ** 2.0)
        v_eff = true_airspeed * factor
        v_tip = np.pi * rps * diameter

        vso = AtmosphereWithPartials(altitude, altitude_in_feet=False).speed_of_sound

        tip_mach = np.sqrt(v_eff**2.0 + v_tip**2.0) / vso
        advance_ratio = v_eff / (rps * diameter)

        return tip_mach, advance_ratio, v_eff, v_tip, rps, factor, vso, diameter, k_duct, alpha, \
            true_airspeed

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        tip_mach, advance_ratio, *_ = self._tip_mach_and_advance_ratio(inputs)

        outputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":tip_mach_max"
        ] = np.max(tip_mach)
        outputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":advance_ratio_max"
        ] = np.max(advance_ratio)
        outputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":rpm_max"
        ] = np.max(inputs["rpm"])
        outputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":torque_max"
        ] = np.max(inputs["torque_in"])
        outputs["advance_ratio"] = advance_ratio

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        (
            tip_mach,
            advance_ratio,
            v_eff,
            v_tip,
            rps,
            factor,
            vso,
            diameter,
            k_duct,
            alpha,
            true_airspeed,
        ) = self._tip_mach_and_advance_ratio(inputs)

        i_mach = np.argmax(tip_mach)
        i_j = np.argmax(advance_ratio)
        i_rpm = np.argmax(inputs["rpm"])

        atm = AtmosphereWithPartials(inputs["altitude"], altitude_in_feet=False)
        d_vso_d_alt = atm.partial_speed_of_sound_altitude

        safe_factor = np.where(factor == 0.0, 1e-12, factor)
        d_factor_d_alpha = np.sin(alpha) * np.cos(alpha) * (k_duct**2.0 - 1.0) / safe_factor
        d_factor_d_kduct = k_duct * np.sin(alpha) ** 2.0 / safe_factor

        # ── d(tip_mach)/d(*), evaluated only at the argmax point ────────────
        d_tipmach_d_vtip = v_tip / (vso**2.0 * tip_mach)
        d_tipmach_d_veff = v_eff / (vso**2.0 * tip_mach)

        d_tipmach_d_rpm = np.zeros(n)
        d_tipmach_d_rpm[i_mach] = d_tipmach_d_vtip[i_mach] * np.pi * diameter / 60.0

        d_tipmach_d_va = np.zeros(n)
        d_tipmach_d_va[i_mach] = d_tipmach_d_veff[i_mach] * factor[i_mach]

        d_tipmach_d_alpha = np.zeros(n)
        d_tipmach_d_alpha[i_mach] = (
            d_tipmach_d_veff[i_mach] * true_airspeed[i_mach] * d_factor_d_alpha[i_mach]
        )

        d_tipmach_d_altitude = np.zeros(n)
        d_tipmach_d_altitude[i_mach] = -tip_mach[i_mach] / vso[i_mach] * d_vso_d_alt[i_mach]

        d_tipmach_d_d = d_tipmach_d_vtip[i_mach] * np.pi * rps[i_mach]
        d_tipmach_d_kduct = d_tipmach_d_veff[i_mach] * true_airspeed[i_mach] * d_factor_d_kduct[i_mach]

        # ── d(advance_ratio)/d(*), evaluated only at its argmax point ──────
        d_j_d_rpm = np.zeros(n)
        d_j_d_rpm[i_j] = -v_eff[i_j] / (rps[i_j] ** 2.0 * diameter) / 60.0

        d_j_d_va = np.zeros(n)
        d_j_d_va[i_j] = factor[i_j] / (rps[i_j] * diameter)

        d_j_d_alpha = np.zeros(n)
        d_j_d_alpha[i_j] = true_airspeed[i_j] / (rps[i_j] * diameter) * d_factor_d_alpha[i_j]

        d_j_d_d = -v_eff[i_j] / (rps[i_j] * diameter**2.0)
        d_j_d_kduct = true_airspeed[i_j] / (rps[i_j] * diameter) * d_factor_d_kduct[i_j]

        # ── d(advance_ratio)/d(*), FULL per-point vectors (not just at the argmax) ──────────
        # Same formulas as d_j_d_rpm/d_j_d_va/d_j_d_alpha/d_j_d_d/d_j_d_kduct above, just kept for
        # every mission point instead of zeroed out except at i_j -- needed because "advance_ratio"
        # is a per-point output, not a scalar reduction like advance_ratio_max.
        d_j_full_d_rpm = -v_eff / (rps**2.0 * diameter) / 60.0
        d_j_full_d_va = factor / (rps * diameter)
        d_j_full_d_alpha = true_airspeed / (rps * diameter) * d_factor_d_alpha
        d_j_full_d_d = -v_eff / (rps * diameter**2.0)
        d_j_full_d_kduct = true_airspeed / (rps * diameter) * d_factor_d_kduct

        # ── d(rpm_max)/d(rpm) ────────────────────────────────────────────────
        d_rpmmax_d_rpm = np.zeros(n)
        d_rpmmax_d_rpm[i_rpm] = 1.0

        # ── d(torque_max)/d(torque_in) ──────────────────────────────────────
        i_torque = np.argmax(inputs["torque_in"])
        d_torquemax_d_torque = np.zeros(n)
        d_torquemax_d_torque[i_torque] = 1.0

        tip_mach_max = "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":tip_mach_max"
        advance_ratio_max = (
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":advance_ratio_max"
        )
        rpm_max = "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":rpm_max"
        torque_max = "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":torque_max"

        partials[tip_mach_max, "rpm"] = d_tipmach_d_rpm
        partials[tip_mach_max, "true_airspeed"] = d_tipmach_d_va
        partials[tip_mach_max, "altitude"] = d_tipmach_d_altitude
        partials[tip_mach_max, "alpha"] = d_tipmach_d_alpha
        partials[tip_mach_max, "torque_in"] = np.zeros(n)
        partials[
            tip_mach_max, "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ] = d_tipmach_d_d
        partials[
            tip_mach_max,
            "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
        ] = d_tipmach_d_kduct

        partials[advance_ratio_max, "rpm"] = d_j_d_rpm
        partials[advance_ratio_max, "true_airspeed"] = d_j_d_va
        partials[advance_ratio_max, "altitude"] = np.zeros(n)
        partials[advance_ratio_max, "alpha"] = d_j_d_alpha
        partials[advance_ratio_max, "torque_in"] = np.zeros(n)
        partials[
            advance_ratio_max,
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
        ] = d_j_d_d
        partials[
            advance_ratio_max,
            "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
        ] = d_j_d_kduct

        partials[rpm_max, "rpm"] = d_rpmmax_d_rpm
        partials[rpm_max, "true_airspeed"] = np.zeros(n)
        partials[rpm_max, "altitude"] = np.zeros(n)
        partials[rpm_max, "alpha"] = np.zeros(n)
        partials[rpm_max, "torque_in"] = np.zeros(n)
        partials[
            rpm_max, "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ] = 0.0
        partials[
            rpm_max, "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct"
        ] = 0.0

        partials[torque_max, "rpm"] = np.zeros(n)
        partials[torque_max, "true_airspeed"] = np.zeros(n)
        partials[torque_max, "altitude"] = np.zeros(n)
        partials[torque_max, "alpha"] = np.zeros(n)
        partials[torque_max, "torque_in"] = d_torquemax_d_torque
        partials[
            torque_max, "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ] = 0.0
        partials[
            torque_max, "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct"
        ] = 0.0

        # ── d(advance_ratio)/d(*), per-point (see declare_partials above) ───────────────────
        partials["advance_ratio", "rpm"] = d_j_full_d_rpm
        partials["advance_ratio", "true_airspeed"] = d_j_full_d_va
        partials["advance_ratio", "alpha"] = d_j_full_d_alpha
        partials[
            "advance_ratio",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
        ] = d_j_full_d_d
        partials[
            "advance_ratio",
            "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
        ] = d_j_full_d_kduct

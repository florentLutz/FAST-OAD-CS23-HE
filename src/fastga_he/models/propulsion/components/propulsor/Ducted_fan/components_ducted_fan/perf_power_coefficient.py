# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
perf_power_coefficient.py
==========================

Takes the "rpm" solved by PerformancesRPM (perf_rpm.py) plus the flight
condition and rotor geometry, and predicts the power coefficient CP with
the same SMT KRG surrogate used for CT. Gradients via the PyTorch MLP
surrogate, same approximation as perf_rpm.py (dCP/dMtip chain not
propagated -- see that module's docstring for the rationale).
"""

import os

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials

from .perf_rpm import load_smt, load_pytorch, J_MIN, J_MAX

# Same default surrogate path pattern as perf_rpm.py -- see that module for the rationale.
# perf_ducted_fan_new.py's default is the one that actually reaches this class (passed explicitly
# as a constructor kwarg); this is for robustness if instantiated standalone.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SURROGATE_PKL = os.path.join(_HERE, "surrogate_smt.pkl")
_DEFAULT_GRAD_SURROGATE_PKL = os.path.join(_HERE, "surrogate_pt.pkl")


class PerformancesPowerCoefficient(om.ExplicitComponent):
    """Computation of the power coefficient of the ducted fan from the converged rpm."""

    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "surrogate_pkl",
            default=_DEFAULT_SURROGATE_PKL,
            desc="Path to the SMT KRG surrogate (.pkl)",
        )
        self.options.declare(
            "grad_surrogate_pkl",
            default=_DEFAULT_GRAD_SURROGATE_PKL,
            desc="Path to the PyTorch MLP surrogate (.pkl)",
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        self._smt_ct, self._smt_cp, self._scaler = load_smt(self.options["surrogate_pkl"])
        self._pt_ct, self._pt_cp = load_pytorch(self.options["grad_surrogate_pkl"])

        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the ducted fan",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades",
            val=np.nan,
            desc="Number of blades on the ducted fan rotor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord",
            val=np.nan,
            units="m",
            desc="Blade chord of the ducted fan rotor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":beta_ref",
            val=0.0,
            desc="Reference blade pitch angle of the ducted fan rotor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":cb_ratio",
            val=0.30,
            desc="Chord to blade ratio parameter used by the aerodynamic surrogate",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
            val=0.4,
            desc="Duct rectification factor of the oblique inflow correction",
        )

        self.add_input("rpm", units="min**-1", val=np.nan, shape=n)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=n)
        self.add_input("altitude", units="m", val=np.nan, shape=n)
        self.add_input("alpha", units="rad", val=0.0, shape=n, desc="Local flow incidence angle")

        self.add_output("power_coefficient", val=0.1, shape=n)

        self.declare_partials(
            of="power_coefficient",
            wrt=[
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades",
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord",
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":beta_ref",
                "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":cb_ratio",
                "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
            ],
            method="exact",
            rows=np.arange(n),
            cols=np.zeros(n),
        )
        self.declare_partials(
            of="power_coefficient",
            wrt=["rpm", "true_airspeed", "altitude", "alpha"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ][0]
        number_blades = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades"
        ][0]
        blade_chord = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord"
        ][0]
        beta_ref = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":beta_ref"
        ][0]
        cb_ratio = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":cb_ratio"
        ][0]
        k_duct = inputs[
            "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct"
        ][0]

        rpm = inputs["rpm"]
        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]
        alpha = inputs["alpha"]

        radius = diameter / 2.0
        sigma = number_blades * blade_chord / (np.pi * radius)

        rps = rpm / 60.0
        factor = np.sqrt(np.cos(alpha) ** 2.0 + k_duct**2.0 * np.sin(alpha) ** 2.0)
        v_eff = true_airspeed * factor
        j = np.clip(v_eff / (rps * diameter), J_MIN, J_MAX)
        v_tip = np.pi * rps * diameter

        vso = AtmosphereWithPartials(altitude, altitude_in_feet=False).speed_of_sound
        m_tip = np.minimum(np.sqrt(v_eff**2.0 + v_tip**2.0) / vso, 0.80)

        x_sc = self._scaler.transform(
            np.column_stack(
                [j, m_tip, np.full(n, sigma), np.full(n, cb_ratio), np.full(n, beta_ref)]
            )
        )
        cp = np.maximum(self._smt_cp.predict_values(x_sc)[:, 0], 1e-6)

        outputs["power_coefficient"] = cp

        self._cache = dict(
            diameter=diameter,
            number_blades=number_blades,
            blade_chord=blade_chord,
            sigma=sigma,
            cb_ratio=cb_ratio,
            beta_ref=beta_ref,
            k_duct=k_duct,
            rps=rps,
            j=j,
            true_airspeed=true_airspeed,
            alpha=alpha,
            x_raw=np.column_stack(
                [j, m_tip, np.full(n, sigma), np.full(n, cb_ratio), np.full(n, beta_ref)]
            ),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        import torch

        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]
        s = self._cache

        diameter = s["diameter"]
        rps = s["rps"]

        d_cp_dj = np.zeros(n)
        d_cp_dsigma = np.zeros(n)
        d_cp_dcb = np.zeros(n)
        d_cp_dbeta = np.zeros(n)

        # dCP/dMtip intentionally not used, same approximation as perf_rpm.py.
        for i in range(n):
            x = torch.tensor([s["x_raw"][i]], dtype=torch.float32, requires_grad=True)
            self._pt_cp(x).backward()
            grad = x.grad.numpy().flatten()
            d_cp_dj[i] = grad[0]
            d_cp_dsigma[i] = grad[2]
            d_cp_dcb[i] = grad[3]
            d_cp_dbeta[i] = grad[4]

        d_j_d_rpm = -s["true_airspeed"] * np.sqrt(
            np.cos(s["alpha"]) ** 2.0 + s["k_duct"] ** 2.0 * np.sin(s["alpha"]) ** 2.0
        ) / (rps**2.0 * diameter) / 60.0
        d_j_d_diameter = -s["true_airspeed"] * np.sqrt(
            np.cos(s["alpha"]) ** 2.0 + s["k_duct"] ** 2.0 * np.sin(s["alpha"]) ** 2.0
        ) / (rps * diameter**2.0)

        factor = np.sqrt(np.cos(s["alpha"]) ** 2.0 + s["k_duct"] ** 2.0 * np.sin(s["alpha"]) ** 2.0)
        safe_factor = np.where(factor == 0.0, 1e-12, factor)
        d_factor_d_alpha = (
            np.sin(s["alpha"]) * np.cos(s["alpha"]) * (s["k_duct"] ** 2.0 - 1.0) / safe_factor
        )
        d_factor_d_kduct = s["k_duct"] * np.sin(s["alpha"]) ** 2.0 / safe_factor

        d_j_d_true_airspeed = factor / (rps * diameter)
        d_j_d_alpha = s["true_airspeed"] / (rps * diameter) * d_factor_d_alpha
        d_j_d_kduct = s["true_airspeed"] / (rps * diameter) * d_factor_d_kduct

        partials["power_coefficient", "rpm"] = d_cp_dj * d_j_d_rpm
        partials["power_coefficient", "true_airspeed"] = d_cp_dj * d_j_d_true_airspeed
        partials["power_coefficient", "alpha"] = d_cp_dj * d_j_d_alpha
        # altitude only affects Mtip (dropped contribution) -- zero here as well.
        partials["power_coefficient", "altitude"] = np.zeros(n)

        # NOTE: sigma also depends on diameter (d_sigma_d_diameter = -sigma / diameter) -- see
        # the identical note in perf_rpm.py's linearize(). Included here for the same reason.
        d_sigma_d_diameter = -s["sigma"] / diameter

        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter",
        ] = d_cp_dj * d_j_d_diameter + d_cp_dsigma * d_sigma_d_diameter

        radius = diameter / 2.0
        d_sigma_d_nb = s["blade_chord"] / (np.pi * radius)
        d_sigma_d_c = s["number_blades"] / (np.pi * radius)

        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades",
        ] = (d_cp_dsigma * d_sigma_d_nb)
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord",
        ] = (d_cp_dsigma * d_sigma_d_c)
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":cb_ratio",
        ] = d_cp_dcb
        partials[
            "power_coefficient",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":beta_ref",
        ] = d_cp_dbeta
        partials[
            "power_coefficient",
            "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct",
        ] = (d_cp_dj * d_j_d_kduct)

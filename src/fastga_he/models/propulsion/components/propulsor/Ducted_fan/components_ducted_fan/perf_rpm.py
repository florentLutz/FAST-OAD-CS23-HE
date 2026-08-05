# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
perf_rpm.py
===========

Fixed architecture (unchanged from edf_propulsion.py):
    Predict   (apply_nonlinear)  -> SMT KRG     (best accuracy)
    Gradient  (linearize)        -> PyTorch MLP autograd (cheap, exact)

Oblique inflow correction (ducted fan, Gentry et al. 1998), unchanged:
    alpha  : local wing angle of attack [rad], a promoted mission variable
             (aerodynamic equilibrium output), NOT a design variable.
    k_duct : duct rectification factor [-], now a tunable "settings:"
             input (like ..components.perf_advance_ratio's
             effective_advance_ratio), default 0.4 (typical ducted fan).
"""

import os

import numpy as np
import openmdao.api as om

from stdatm import AtmosphereWithPartials

J_MIN = 0.01
J_MAX = 1.20
RPM_MIN = 500.0
RPM_MAX = 30000.0

# Default surrogate paths, same rationale as perf_ducted_fan_new.py: perf_ducted_fan_new.py's own
# default is what actually reaches this class (passed explicitly as a constructor kwarg), so this
# one is just for robustness if PerformancesRPM is ever instantiated directly (e.g. in a unit test).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SURROGATE_PKL = os.path.join(_HERE, "surrogate_smt.pkl")
_DEFAULT_GRAD_SURROGATE_PKL = os.path.join(_HERE, "surrogate_pt.pkl")


def load_smt(pkl_path):
    """Load SMT KRG surrogate. Used for predict (best accuracy)."""
    import joblib

    p = joblib.load(pkl_path)
    assert p.get("tipo") == "SMT_KRG", "surrogate_pkl: expected SMT_KRG, got '%s'" % p.get("tipo")
    return p["modelo_ct"], p["modelo_cp"], p["scaler"]


def load_pytorch(pkl_path):
    """
    Reconstruct PyTorch MLP from pkl. Used for gradients (cheap, exact).
    Returns (net_ct, net_cp), both warmed up and in eval mode.
    """
    import joblib
    import torch
    import torch.nn as nn

    p = joblib.load(pkl_path)
    assert p.get("tipo") == "PYTORCH_MLP", (
        "grad_surrogate_pkl: expected PYTORCH_MLP, got '%s'" % p.get("tipo")
    )

    class Net(nn.Module):
        def __init__(self, mean_x, std_x, hidden):
            super().__init__()
            self.register_buffer("mean_X", torch.tensor(mean_x, dtype=torch.float32))
            self.register_buffer("std_X", torch.tensor(std_x, dtype=torch.float32))
            layers = []
            in_d = 5
            for out_d in hidden:
                layers += [nn.Linear(in_d, out_d), nn.Tanh()]
                in_d = out_d
            layers += [nn.Linear(in_d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net((x - self.mean_X) / self.std_X).squeeze(-1)

    hidden = p.get("hidden", (128, 64, 64, 32))
    ct = Net(p["mean_X"], p["std_X"], hidden)
    cp = Net(p["mean_X"], p["std_X"], hidden)
    ct.load_state_dict(p["state_ct"])
    ct.eval()
    cp.load_state_dict(p["state_cp"])
    cp.eval()

    with torch.no_grad():
        _warm = torch.zeros(1, 5)
        ct(_warm)
        cp(_warm)

    return ct, cp


class PerformancesRPM(om.ImplicitComponent):
    """
    Residual:  F(rpm) = CT * density * (rpm/60)^2 * diameter^4 - thrust = 0

    Predict via SMT KRG (batched over all mission points). Gradients via
    PyTorch autograd + chain rule (looped per point, exact).
    """

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
            desc="Duct rectification factor of the oblique inflow correction (0=perfect duct, "
            "1=open propeller)",
        )

        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=n)
        self.add_input("altitude", units="m", val=np.nan, shape=n)
        self.add_input("density", units="kg/m**3", val=np.nan, shape=n)
        self.add_input("alpha", units="rad", val=0.0, shape=n, desc="Local flow incidence angle")
        self.add_input("thrust", units="N", val=np.nan, shape=n)

        self.add_output(
            "rpm", val=12000.0, units="min**-1", shape=n, lower=RPM_MIN, upper=RPM_MAX
        )

    def setup_partials(self):
        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]

        self.declare_partials(
            of="rpm",
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
            of="rpm",
            wrt=["rpm", "true_airspeed", "altitude", "density", "alpha", "thrust"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )

    def _kinematics(self, rpm, diameter, true_airspeed, alpha, k_duct):
        """Rotational state + flight condition -> surrogate inputs (J, Mtip-related V_eff)."""
        rps = rpm / 60.0

        factor = np.sqrt(np.cos(alpha) ** 2.0 + k_duct**2.0 * np.sin(alpha) ** 2.0)
        v_eff = true_airspeed * factor

        j = np.clip(v_eff / (rps * diameter), J_MIN, J_MAX)
        v_tip = np.pi * rps * diameter

        return j, v_tip, v_eff, factor, rps

    def apply_nonlinear(self, inputs, outputs, residuals):
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

        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]
        density = inputs["density"]
        alpha = inputs["alpha"]
        thrust = inputs["thrust"]
        rpm = outputs["rpm"]

        radius = diameter / 2.0
        sigma = number_blades * blade_chord / (np.pi * radius)

        j, v_tip, v_eff, factor, rps = self._kinematics(
            rpm, diameter, true_airspeed, alpha, k_duct
        )

        vso = AtmosphereWithPartials(altitude, altitude_in_feet=False).speed_of_sound
        m_tip = np.minimum(np.sqrt(v_eff**2.0 + v_tip**2.0) / vso, 0.80)

        x_sc = self._scaler.transform(
            np.column_stack(
                [j, m_tip, np.full(n, sigma), np.full(n, cb_ratio), np.full(n, beta_ref)]
            )
        )
        ct = np.maximum(self._smt_ct.predict_values(x_sc)[:, 0], 1e-6)

        residuals["rpm"] = ct * density * rps**2.0 * diameter**4.0 - thrust

        self._cache = dict(
            diameter=diameter,
            number_blades=number_blades,
            blade_chord=blade_chord,
            sigma=sigma,
            cb_ratio=cb_ratio,
            beta_ref=beta_ref,
            k_duct=k_duct,
            true_airspeed=true_airspeed,
            alpha=alpha,
            density=density,
            rps=rps,
            j=j,
            m_tip=m_tip,
            ct=ct,
            x_raw=np.column_stack(
                [j, m_tip, np.full(n, sigma), np.full(n, cb_ratio), np.full(n, beta_ref)]
            ),
        )

    def linearize(self, inputs, outputs, jacobian):
        import torch

        ducted_fan_id = self.options["ducted_fan_id"]
        n = self.options["number_of_points"]
        s = self._cache

        diameter = s["diameter"]
        rps = s["rps"]
        density = s["density"]
        ct = s["ct"]
        v_eff = s["true_airspeed"] * np.sqrt(
            np.cos(s["alpha"]) ** 2.0 + s["k_duct"] ** 2.0 * np.sin(s["alpha"]) ** 2.0
        )

        scale_f = density * rps**2.0 * diameter**4.0

        d_ct_dj = np.zeros(n)
        d_ct_dsigma = np.zeros(n)
        d_ct_dcb = np.zeros(n)
        d_ct_dbeta = np.zeros(n)

        # dCT/dMtip is intentionally NOT used -- see module docstring (approximation carried
        # over from edf_propulsion.py).
        for i in range(n):
            x = torch.tensor([s["x_raw"][i]], dtype=torch.float32, requires_grad=True)
            self._pt_ct(x).backward()
            grad = x.grad.numpy().flatten()
            d_ct_dj[i] = grad[0]
            d_ct_dsigma[i] = grad[2]
            d_ct_dcb[i] = grad[3]
            d_ct_dbeta[i] = grad[4]

        d_j_d_rpm = -v_eff / (rps**2.0 * diameter) / 60.0
        d_j_d_diameter = -v_eff / (rps * diameter**2.0)

        factor = np.sqrt(
            np.cos(s["alpha"]) ** 2.0 + s["k_duct"] ** 2.0 * np.sin(s["alpha"]) ** 2.0
        )
        # Avoid division by zero when alpha == 0 and k_duct == 1 (factor == 1, fine); guard
        # against factor == 0, which would require cos(alpha) = 0 and k_duct = 0.
        safe_factor = np.where(factor == 0.0, 1e-12, factor)
        d_factor_d_alpha = (
            np.sin(s["alpha"]) * np.cos(s["alpha"]) * (s["k_duct"] ** 2.0 - 1.0) / safe_factor
        )
        d_factor_d_kduct = s["k_duct"] * np.sin(s["alpha"]) ** 2.0 / safe_factor

        d_j_d_true_airspeed = factor / (rps * diameter)
        d_j_d_alpha = s["true_airspeed"] / (rps * diameter) * d_factor_d_alpha
        d_j_d_kduct = s["true_airspeed"] / (rps * diameter) * d_factor_d_kduct

        # ── state self-partial: d(rpm residual)/d(rpm) ───────────────────────
        jacobian["rpm", "rpm"] = d_ct_dj * d_j_d_rpm * scale_f + ct * density * diameter**4.0 * (
            2.0 * rps / 60.0
        )

        # ── vectorized (per-point) inputs ─────────────────────────────────────
        jacobian["rpm", "true_airspeed"] = d_ct_dj * d_j_d_true_airspeed * scale_f
        jacobian["rpm", "alpha"] = d_ct_dj * d_j_d_alpha * scale_f
        jacobian["rpm", "density"] = ct * rps**2.0 * diameter**4.0
        jacobian["rpm", "thrust"] = -np.ones(n)
        # altitude only affects Mtip (via speed of sound), whose contribution to CT is
        # intentionally dropped (see module docstring) -- so its partial is zero here.
        jacobian["rpm", "altitude"] = np.zeros(n)

        # ── scalar (broadcast) inputs ──────────────────────────────────────────
        # NOTE: sigma = number_blades * blade_chord / (pi * diameter / 2) also depends on
        # diameter (d_sigma_d_diameter = -sigma / diameter) -- this chain term through
        # dCT/dsigma was missing in edf_propulsion.py's original _adim()/linearize() (it only
        # returned/used dsigma_dNbl and dsigma_dc), which under-estimated the diameter partial.
        # Fixed here; confirmed against finite-difference during conversion testing.
        d_sigma_d_diameter = -s["sigma"] / diameter

        d_f_d_d = (
            d_ct_dj * d_j_d_diameter * scale_f
            + d_ct_dsigma * d_sigma_d_diameter * scale_f
            + ct * density * rps**2.0 * 4.0 * diameter**3.0
        )
        jacobian[
            "rpm", "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ] = d_f_d_d

        radius = diameter / 2.0
        d_sigma_d_nb = s["blade_chord"] / (np.pi * radius)
        d_sigma_d_c = s["number_blades"] / (np.pi * radius)

        jacobian[
            "rpm", "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":number_blades"
        ] = (d_ct_dsigma * scale_f * d_sigma_d_nb)
        jacobian[
            "rpm", "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":blade_chord"
        ] = (d_ct_dsigma * scale_f * d_sigma_d_c)
        jacobian[
            "rpm", "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":cb_ratio"
        ] = (d_ct_dcb * scale_f)
        jacobian[
            "rpm", "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":beta_ref"
        ] = (d_ct_dbeta * scale_f)
        jacobian[
            "rpm", "settings:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":k_duct"
        ] = (d_ct_dj * d_j_d_kduct * scale_f)

    def guess_nonlinear(self, inputs, outputs, residuals):
        ducted_fan_id = self.options["ducted_fan_id"]

        diameter = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":diameter"
        ][0]
        true_airspeed = np.maximum(inputs["true_airspeed"], 1.0)

        outputs["rpm"] = true_airspeed / (0.4 * diameter) * 60.0

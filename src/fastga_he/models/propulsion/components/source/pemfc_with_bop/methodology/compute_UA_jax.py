"""
CalculateUA - OpenMDAO ExplicitComponent
========================================
Computes the real overall heat transfer coefficient-area product (UA_real)
for a plate-fin heat exchanger, given the cold-side (L_cold) and hot-side
(L_hot) flow-path lengths.

Derivatives are computed analytically via JAX reverse-mode AD and fed into
OpenMDAO through `compute_partials` (Strategy 3).

Design decisions
----------------
* All physics are encoded in `_calculate_UA_jax`, a pure JAX function that
  operates on scalar jnp arrays.  No Python control-flow that depends on
  traced values is allowed inside that function; conditional branches on Re
  are replaced with `jnp.where`.
* The outer `if` that selects between the two fin-correlation families
  (specific geometry vs. general Chang-Wang) IS kept as a plain Python `if`
  because it depends only on fixed geometry settings, not on optimisation
  variables.  JAX traces the chosen branch at setup time and JIT-compiles it.
* `jax.jit` is applied once (lazily, on first call) to both the forward
  function and its Jacobian so that repeated evaluations inside an OpenMDAO
  driver loop are fast.
* The returned Jacobian has shape (n_outputs, n_inputs) where
    outputs = [UA_real, G_cold, G_hot, f_cold, f_hot, sigma_cold, sigma_hot]
    inputs  = [L_cold, L_hot,  ... all scalar geometry/fluid parameters ...]
  Only the rows/columns that OpenMDAO cares about are extracted.

Dependencies
------------
    pip install jax jaxlib openmdao
"""

import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om

# Enable 64-bit precision so JAX matches NumPy's default dtype and avoids
# round-off differences vs. the original SciPy-based implementation.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Pure JAX physics kernel
# ---------------------------------------------------------------------------


def _make_calculate_UA_jax(use_specific_geometry: bool):
    """
    Return a JAX-traceable function `calculate_UA_jax(L_cold, L_hot, p)`
    where `p` is a dict of frozen scalar parameters.

    Two variants are compiled at import time (one per fin-correlation family)
    and cached as JIT-compiled callables.

    Parameters
    ----------
    use_specific_geometry : bool
        True  → use the tabulated j/f correlations for the specific surface
                 (b_plate = 6.35 mm, fin_area_ratio = 0.809, beta = 1548 1/m).
        False → use the general Chang-Wang (1997) offset-strip correlations.
    """

    def _j_f(Re, alpha_fin, delta_fin, gamma_fin):
        """Colburn j-factor and Fanning f-factor (same correlation for both sides)."""
        if use_specific_geometry:
            j = jnp.where(Re < 1500.0, 0.52 * Re ** (-0.51), 0.41 * Re ** (-0.46))
            f = jnp.where(Re < 1500.0, 6.04 * Re ** (-0.68), 0.36 * Re ** (-0.28))
        else:
            j = (
                0.6522
                * Re ** (-0.5403)
                * alpha_fin ** (-0.1541)
                * delta_fin ** (0.1499)
                * gamma_fin ** (-0.0678)
                * (
                    1.0
                    + 5.259e-5
                    * Re**1.340
                    * alpha_fin**0.504
                    * delta_fin**0.456
                    * gamma_fin ** (-1.055)
                )
                ** 0.1
            )
            f = (
                9.6243
                * Re ** (-0.7422)
                * alpha_fin ** (-0.1856)
                * delta_fin ** (0.3053)
                * gamma_fin ** (-0.2659)
                * (
                    1.0
                    + 7.669e-8
                    * Re**4.429
                    * alpha_fin**0.920
                    * delta_fin**3.767
                    * gamma_fin ** (0.236)
                )
                ** 0.1
            )
        return j, f

    def calculate_UA_jax(L_cold, L_hot, p):
        """
        Compute UA_real and ancillary outputs.

        Parameters
        ----------
        L_cold : jnp scalar  – cold-side flow-path length [m]
        L_hot  : jnp scalar  – hot-side  flow-path length [m]
        p      : dict of jnp scalars – frozen geometric and fluid parameters
                 (see CalculateUA.setup for the full list)

        Returns
        -------
        (UA_real, G_cold, G_hot, f_cold, f_hot, sigma_cold, sigma_hot)
        All scalars.
        """
        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        height = (
            (p["n_cold"] + p["n_hot"]) * (p["h_fin"] + p["t_fin"])
            + p["n_sep"] * p["t_plate"]
            + 2.0 * p["t_plate_end"]
        )
        V = L_cold * L_hot * height

        n_layers = p["n_cold"] + p["n_hot"]
        A_plate = (n_layers - 1.0) * L_cold * L_hot

        # ------------------------------------------------------------------
        # Cold side (air)
        # ------------------------------------------------------------------
        A_cold = p["alpha"] * V
        A_fr_cold = L_hot * height
        sigma_cold = p["alpha"] * p["d_h"] / 4.0
        A_ff_cold = sigma_cold * A_fr_cold
        G_cold = p["m_cold"] / A_ff_cold
        Re_cold = p["d_h"] * G_cold / p["mu_cold"]

        # ------------------------------------------------------------------
        # Hot side (coolant)
        # ------------------------------------------------------------------
        A_hot = p["alpha"] * V
        A_fr_hot = L_cold * height
        sigma_hot = p["alpha"] * p["d_h"] / 4.0
        A_ff_hot = sigma_hot * A_fr_hot
        G_hot = p["m_hot"] / A_ff_hot
        Re_hot = p["d_h"] * G_hot / p["mu_hot"]

        # ------------------------------------------------------------------
        # Fin geometry ratios
        # ------------------------------------------------------------------
        alpha_fin = p["s_fin"] / p["h_fin"]
        delta_fin = p["t_fin"] / p["l_fin"]
        gamma_fin = p["t_fin"] / p["s_fin"]

        # ------------------------------------------------------------------
        # j and f factors
        # ------------------------------------------------------------------
        j_cold, f_cold = _j_f(Re_cold, alpha_fin, delta_fin, gamma_fin)
        j_hot, f_hot = _j_f(Re_hot, alpha_fin, delta_fin, gamma_fin)

        # ------------------------------------------------------------------
        # Heat-transfer coefficients and fin efficiencies
        # ------------------------------------------------------------------
        h_cold = j_cold * Re_cold * p["Pr_cold"] ** (1.0 / 3.0) * p["k_cold"] / p["d_h"]
        m_fin_c = jnp.sqrt(2.0 * h_cold / p["k_fin"] / p["t_fin"])
        nf_cold = jnp.tanh(m_fin_c * p["h_fin"] / 2.0) / (m_fin_c * p["h_fin"] / 2.0)
        no_cold = 1.0 - p["fin_ratio"] * (1.0 - nf_cold)

        h_hot = j_hot * Re_hot * p["Pr_hot"] ** (1.0 / 3.0) * p["k_hot"] / p["d_h"]
        m_fin_h = jnp.sqrt(2.0 * h_hot / p["k_fin"] / p["t_fin"])
        nf_hot = jnp.tanh(m_fin_h * p["h_fin"] / 2.0) / (m_fin_h * p["h_fin"] / 2.0)
        no_hot = 1.0 - p["fin_ratio"] * (1.0 - nf_hot)

        # ------------------------------------------------------------------
        # Overall UA
        # ------------------------------------------------------------------
        UA_real = 1.0 / (
            1.0 / (no_cold * h_cold * A_cold)
            + p["t_plate"] / p["k_w"] / A_plate
            + 1.0 / (no_hot * h_hot * A_hot)
        )

        return UA_real, G_cold, G_hot, f_cold, f_hot, sigma_cold, sigma_hot

    return calculate_UA_jax


# ---------------------------------------------------------------------------
# OpenMDAO component
# ---------------------------------------------------------------------------


class CalculateUA(om.ExplicitComponent):
    """
    Computes UA_real (and associated flow/friction quantities) for a
    cross-flow plate-fin HEX given the two flow-path lengths.

    Fluid properties (rho, cp, mu, Pr, k) are expected to be pre-evaluated
    upstream (e.g. by a FluidProperties component) and passed as inputs so
    that this component remains fully differentiable via JAX.

    Analytic partial derivatives are provided through `compute_partials`
    using `jax.jacobian` (Strategy 3 — external JAX, OpenMDAO partials).
    """

    def setup(self):
        # ------------------------------------------------------------------
        # Design variables (optimised quantities)
        # ------------------------------------------------------------------
        self.add_input("data:thermal:HEX_ambient:length", units="m", val=0.1)  # L_cold
        self.add_input("data:thermal:HEX_ambient:width", units="m", val=0.1)  # L_hot

        # ------------------------------------------------------------------
        # Fluid properties – cold side (atmospheric air after diffuser)
        # ------------------------------------------------------------------
        self.add_input("data:thermal:cold_side:mass_flow", units="kg/s", val=np.nan)
        self.add_input("data:thermal:cold_side:dynamic_viscosity", units="kg/m/s", val=np.nan)
        self.add_input("data:thermal:cold_side:Prandtl", val=np.nan)
        self.add_input("data:thermal:cold_side:thermal_conductivity", units="W/m/K", val=np.nan)

        # ------------------------------------------------------------------
        # Fluid properties – hot side (coolant)
        # ------------------------------------------------------------------
        self.add_input("data:thermal:hot_side:mass_flow", units="kg/s", val=np.nan)
        self.add_input("data:thermal:hot_side:dynamic_viscosity", units="kg/m/s", val=np.nan)
        self.add_input("data:thermal:hot_side:Prandtl", val=np.nan)
        self.add_input("data:thermal:hot_side:thermal_conductivity", units="W/m/K", val=np.nan)

        # ------------------------------------------------------------------
        # HEX geometry (fixed settings, not differentiated w.r.t. design)
        # ------------------------------------------------------------------
        self.add_input("settings:thermal:HEX:geometry:fin:conductivity", units="W/m/K", val=237.0)
        self.add_input("settings:thermal:HEX:geometry:fin:thickness", units="m", val=0.102e-3)
        self.add_input("settings:thermal:HEX:geometry:fin:frequency", units="1/m", val=782.0)
        self.add_input("settings:thermal:HEX:geometry:fin:length", units="m", val=3.175e-3)
        self.add_input("settings:thermal:HEX:geometry:hydraulic_diameter", units="m", val=2.38e-3)
        self.add_input(
            "settings:thermal:HEX:geometry:surface_area:density", units="1/m", val=2254.0
        )
        self.add_input("settings:thermal:HEX:geometry:fin_area_ratio_to_total_area", val=0.785)
        self.add_input("settings:thermal:HEX:geometry:plate:thickness", units="m", val=0.8e-3)
        self.add_input("settings:thermal:HEX:geometry:plate:spacing", units="m", val=6.35e-3)
        self.add_input("settings:thermal:HEX:geometry:plate:conductivity", units="W/m/K", val=237.0)
        self.add_input("settings:thermal:HEX:geometry:plate:end_thickness", units="m", val=4.4e-3)
        self.add_input("settings:thermal:HEX_ambient:number_of_cold_layers", val=np.nan)

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        self.add_output("data:thermal:HEX_ambient:UA_real", units="W/K")
        self.add_output("data:thermal:HEX_ambient:cold_side:G", units="kg/m**2/s")
        self.add_output("data:thermal:HEX_ambient:hot_side:G", units="kg/m**2/s")
        self.add_output("data:thermal:HEX_ambient:cold_side:f_friction")
        self.add_output("data:thermal:HEX_ambient:hot_side:f_friction")
        self.add_output("data:thermal:HEX_ambient:cold_side:sigma")
        self.add_output("data:thermal:HEX_ambient:hot_side:sigma")

        # Internal cache: will hold JIT-compiled forward fn and Jacobian fn
        self._jit_fwd = None
        self._jit_jac = None
        self._use_specific_geometry = None  # set on first compute

    def setup_partials(self):
        # Declare dense partials for all outputs w.r.t. all inputs.
        # JAX will compute the actual values in compute_partials.
        self.declare_partials("*", "*", method="exact")

    # ------------------------------------------------------------------
    # Helper: pack inputs into the parameter dict expected by JAX kernel
    # ------------------------------------------------------------------
    @staticmethod
    def _s(inputs, key):
        return float(inputs[key].item())

    def _pack_params(self, inputs):
        s = self._s  # shorthand
        n_cold = s(inputs, "settings:thermal:HEX_ambient:number_of_cold_layers")
        n_hot = n_cold - 1.0
        n_sep = n_cold + n_hot - 1.0

        b_plate = s(inputs, "settings:thermal:HEX:geometry:plate:spacing")
        beta = s(inputs, "settings:thermal:HEX:geometry:surface_area:density")
        t_fin = s(inputs, "settings:thermal:HEX:geometry:fin:thickness")
        f_fin = s(inputs, "settings:thermal:HEX:geometry:fin:frequency")
        fin_ratio = s(inputs, "settings:thermal:HEX:geometry:fin_area_ratio_to_total_area")
        t_plate = s(inputs, "settings:thermal:HEX:geometry:plate:thickness")

        h_fin = b_plate - t_fin
        s_fin = 1.0 / f_fin - t_fin
        alpha = b_plate * beta / (b_plate + b_plate + 2.0 * t_plate)

        p = {
            # Layer counts
            "n_cold": jnp.float64(n_cold),
            "n_hot": jnp.float64(n_hot),
            "n_sep": jnp.float64(n_sep),
            # Fin geometry
            "h_fin": jnp.float64(h_fin),
            "s_fin": jnp.float64(s_fin),
            "t_fin": jnp.float64(t_fin),
            "l_fin": jnp.float64(s(inputs, "settings:thermal:HEX:geometry:fin:length")),
            "alpha": jnp.float64(alpha),
            "fin_ratio": jnp.float64(fin_ratio),
            # Plate geometry
            "t_plate": jnp.float64(t_plate),
            "t_plate_end": jnp.float64(
                s(inputs, "settings:thermal:HEX:geometry:plate:end_thickness")
            ),
            # Hydraulic diameter
            "d_h": jnp.float64(s(inputs, "settings:thermal:HEX:geometry:hydraulic_diameter")),
            # Conductivities
            "k_fin": jnp.float64(s(inputs, "settings:thermal:HEX:geometry:fin:conductivity")),
            "k_w": jnp.float64(s(inputs, "settings:thermal:HEX:geometry:plate:conductivity")),
            # Cold-side fluid
            "m_cold": jnp.float64(s(inputs, "data:thermal:cold_side:mass_flow")),
            "mu_cold": jnp.float64(s(inputs, "data:thermal:cold_side:dynamic_viscosity")),
            "Pr_cold": jnp.float64(s(inputs, "data:thermal:cold_side:Prandtl")),
            "k_cold": jnp.float64(s(inputs, "data:thermal:cold_side:thermal_conductivity")),
            # Hot-side fluid
            "m_hot": jnp.float64(s(inputs, "data:thermal:hot_side:mass_flow")),
            "mu_hot": jnp.float64(s(inputs, "data:thermal:hot_side:dynamic_viscosity")),
            "Pr_hot": jnp.float64(s(inputs, "data:thermal:hot_side:Prandtl")),
            "k_hot": jnp.float64(s(inputs, "data:thermal:hot_side:thermal_conductivity")),
        }
        return p

    # ------------------------------------------------------------------
    # Decide which fin-correlation family and (re)build JIT callables
    # ------------------------------------------------------------------
    def _maybe_rebuild_jit(self, inputs):
        b_plate = self._s(inputs, "settings:thermal:HEX:geometry:plate:spacing")
        fin_ratio = self._s(inputs, "settings:thermal:HEX:geometry:fin_area_ratio_to_total_area")
        beta = self._s(inputs, "settings:thermal:HEX:geometry:surface_area:density")

        use_specific = (
            abs(b_plate - 6.35e-3) < 1e-9
            and abs(fin_ratio - 0.809) < 1e-6
            and abs(beta - 1548.0) < 1e-3
        )

        if use_specific == self._use_specific_geometry and self._jit_fwd is not None:
            return  # already compiled for this branch

        self._use_specific_geometry = use_specific
        raw_fn = _make_calculate_UA_jax(use_specific)

        # Forward: returns a tuple of 7 scalars
        def fwd(L_cold, L_hot, p):
            return raw_fn(L_cold, L_hot, p)

        self._jit_fwd = jax.jit(fwd)

        # Stack outputs into a single array so jax.jacobian gives a clean
        # (n_outputs,) array per differentiated argument — not a tuple-of-tuples.
        def fwd_stacked(L_cold, L_hot, p):
            return jnp.stack(list(raw_fn(L_cold, L_hot, p)))  # shape (7,)

        jit_stacked = jax.jit(fwd_stacked)

        # d(outputs[7]) / d(L_cold, L_hot) → each has shape (7,)
        self._jit_jac_lengths = jax.jit(jax.jacobian(jit_stacked, argnums=(0, 1)))

        # d(outputs[7]) / d(scalar param) → shape (7,)
        self._jit_jac_param = jax.jit(jax.jacobian(jit_stacked, argnums=2))

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------
    def compute(self, inputs, outputs):
        self._maybe_rebuild_jit(inputs)

        L_cold = jnp.float64(self._s(inputs, "data:thermal:HEX_ambient:length"))
        L_hot = jnp.float64(self._s(inputs, "data:thermal:HEX_ambient:width"))
        p = self._pack_params(inputs)

        results = self._jit_fwd(L_cold, L_hot, p)
        UA_real, G_cold, G_hot, f_cold, f_hot, sigma_cold, sigma_hot = [float(r) for r in results]

        outputs["data:thermal:HEX_ambient:UA_real"] = UA_real
        outputs["data:thermal:HEX_ambient:cold_side:G"] = G_cold
        outputs["data:thermal:HEX_ambient:hot_side:G"] = G_hot
        outputs["data:thermal:HEX_ambient:cold_side:f_friction"] = f_cold
        outputs["data:thermal:HEX_ambient:hot_side:f_friction"] = f_hot
        outputs["data:thermal:HEX_ambient:cold_side:sigma"] = sigma_cold
        outputs["data:thermal:HEX_ambient:hot_side:sigma"] = sigma_hot

    # ------------------------------------------------------------------
    # compute_partials  (Strategy 3: JAX-provided analytic derivatives)
    # ------------------------------------------------------------------
    def compute_partials(self, inputs, partials):
        self._maybe_rebuild_jit(inputs)

        L_cold = jnp.float64(self._s(inputs, "data:thermal:HEX_ambient:length"))
        L_hot = jnp.float64(self._s(inputs, "data:thermal:HEX_ambient:width"))
        p = self._pack_params(inputs)

        output_names = [
            "data:thermal:HEX_ambient:UA_real",
            "data:thermal:HEX_ambient:cold_side:G",
            "data:thermal:HEX_ambient:hot_side:G",
            "data:thermal:HEX_ambient:cold_side:f_friction",
            "data:thermal:HEX_ambient:hot_side:f_friction",
            "data:thermal:HEX_ambient:cold_side:sigma",
            "data:thermal:HEX_ambient:hot_side:sigma",
        ]

        # ----------------------------------------------------------------
        # Partials w.r.t. L_cold and L_hot
        #
        # _jit_jac_lengths returns a tuple of TWO arrays (one per argnums),
        # each of shape (7,):  jac[0] = d(outputs)/dL_cold,
        #                      jac[1] = d(outputs)/dL_hot
        # ----------------------------------------------------------------
        jac_L = self._jit_jac_lengths(L_cold, L_hot, p)  # tuple of 2 x (7,)

        for i_inp, inp_name in enumerate(
            [
                "data:thermal:HEX_ambient:length",
                "data:thermal:HEX_ambient:width",
            ]
        ):
            for i_out, out_name in enumerate(output_names):
                partials[out_name, inp_name] = float(jac_L[i_inp][i_out])

        # ----------------------------------------------------------------
        # Partials w.r.t. scalar fluid parameters
        #
        # _jit_jac_param differentiates w.r.t. the whole dict p.
        # JAX returns a dict-of-arrays with the same structure as p,
        # each leaf having shape (7,) — one row per output.
        # ----------------------------------------------------------------
        param_input_map = {
            "data:thermal:cold_side:mass_flow": "m_cold",
            "data:thermal:cold_side:dynamic_viscosity": "mu_cold",
            "data:thermal:cold_side:Prandtl": "Pr_cold",
            "data:thermal:cold_side:thermal_conductivity": "k_cold",
            "data:thermal:hot_side:mass_flow": "m_hot",
            "data:thermal:hot_side:dynamic_viscosity": "mu_hot",
            "data:thermal:hot_side:Prandtl": "Pr_hot",
            "data:thermal:hot_side:thermal_conductivity": "k_hot",
        }

        # One jacobian call returns derivatives w.r.t. every key in p at once
        jac_p = self._jit_jac_param(L_cold, L_hot, p)  # dict of {key: array(7,)}

        for inp_name, p_key in param_input_map.items():
            for i_out, out_name in enumerate(output_names):
                partials[out_name, inp_name] = float(jac_p[p_key][i_out])

        # ----------------------------------------------------------------
        # Geometry settings — fixed, zero partials
        # ----------------------------------------------------------------
        geometry_inputs = [
            "settings:thermal:HEX:geometry:fin:conductivity",
            "settings:thermal:HEX:geometry:fin:thickness",
            "settings:thermal:HEX:geometry:fin:frequency",
            "settings:thermal:HEX:geometry:fin:length",
            "settings:thermal:HEX:geometry:hydraulic_diameter",
            "settings:thermal:HEX:geometry:surface_area:density",
            "settings:thermal:HEX:geometry:fin_area_ratio_to_total_area",
            "settings:thermal:HEX:geometry:plate:thickness",
            "settings:thermal:HEX:geometry:plate:spacing",
            "settings:thermal:HEX:geometry:plate:conductivity",
            "settings:thermal:HEX:geometry:plate:end_thickness",
            "settings:thermal:HEX_ambient:number_of_cold_layers",
        ]
        for inp_name in geometry_inputs:
            for out_name in output_names:
                partials[out_name, inp_name] = 0.0


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import openmdao.api as om

    prob = om.Problem()
    comp = prob.model.add_subsystem("calc_UA", CalculateUA(), promotes=["*"])
    prob.setup(force_alloc_complex=True)

    # -- geometry (default Kays & London surface 'louvered-fin' geometry)
    prob.set_val("settings:thermal:HEX_ambient:number_of_cold_layers", 6)
    prob.set_val("settings:thermal:HEX:geometry:fin:conductivity", 237.0, units="W/m/K")
    prob.set_val("settings:thermal:HEX:geometry:fin:thickness", 0.102e-3, units="m")
    prob.set_val("settings:thermal:HEX:geometry:fin:frequency", 782.0, units="1/m")
    prob.set_val("settings:thermal:HEX:geometry:fin:length", 3.175e-3, units="m")
    prob.set_val("settings:thermal:HEX:geometry:hydraulic_diameter", 2.38e-3, units="m")
    prob.set_val("settings:thermal:HEX:geometry:surface_area:density", 2254.0, units="1/m")
    prob.set_val("settings:thermal:HEX:geometry:fin_area_ratio_to_total_area", 0.785)
    prob.set_val("settings:thermal:HEX:geometry:plate:thickness", 0.8e-3, units="m")
    prob.set_val("settings:thermal:HEX:geometry:plate:spacing", 6.35e-3, units="m")
    prob.set_val("settings:thermal:HEX:geometry:plate:conductivity", 237.0, units="W/m/K")
    prob.set_val("settings:thermal:HEX:geometry:plate:end_thickness", 4.4e-3, units="m")

    # -- design variables
    prob.set_val("data:thermal:HEX_ambient:length", 0.3, units="m")
    prob.set_val("data:thermal:HEX_ambient:width", 0.2, units="m")

    # -- fluid properties (air cold side, water hot side – representative values)
    prob.set_val("data:thermal:cold_side:mass_flow", 1.0, units="kg/s")
    prob.set_val("data:thermal:cold_side:dynamic_viscosity", 1.85e-5, units="kg/m/s")
    prob.set_val("data:thermal:cold_side:Prandtl", 0.713)
    prob.set_val("data:thermal:cold_side:thermal_conductivity", 0.026, units="W/m/K")

    prob.set_val("data:thermal:hot_side:mass_flow", 0.5, units="kg/s")
    prob.set_val("data:thermal:hot_side:dynamic_viscosity", 8.9e-4, units="kg/m/s")
    prob.set_val("data:thermal:hot_side:Prandtl", 6.13)
    prob.set_val("data:thermal:hot_side:thermal_conductivity", 0.606, units="W/m/K")

    prob.run_model()

    print("UA_real   =", prob.get_val("data:thermal:HEX_ambient:UA_real"), "W/K")
    print("G_cold    =", prob.get_val("data:thermal:HEX_ambient:cold_side:G"), "kg/m²/s")
    print("G_hot     =", prob.get_val("data:thermal:HEX_ambient:hot_side:G"), "kg/m²/s")
    print("sigma_cold=", prob.get_val("data:thermal:HEX_ambient:cold_side:sigma"))
    print("sigma_hot =", prob.get_val("data:thermal:HEX_ambient:hot_side:sigma"))

    # Verify analytic partials against complex-step
    data = prob.check_partials(compact_print=True)
    print("\nPartial check complete.")

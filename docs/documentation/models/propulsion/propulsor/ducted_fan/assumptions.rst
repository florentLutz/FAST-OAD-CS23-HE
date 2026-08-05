.. _assumptions-ducted-fan:

=============================
Ducted fan model assumptions
=============================

The following assumptions have been made in the calculation of the ducted fan model.

* The rotor and duct mass are estimated from an empirical fit to two commercial ducted fan units;
  blade chord is not used in the mass formula (no reliable calibration data for it).
* The gradient of the thrust/power coefficients with respect to tip Mach number is not propagated
  through the rpm solve -- only the advance-ratio and blade-geometry chains are used.
* The duct's external drag is built up from a classic nacelle friction/form-factor model evaluated
  at a single representative velocity/altitude per flight regime (low-speed, cruise), not the
  actual mission profile -- an accepted simplification since the friction coefficient is only
  weakly sensitive to Reynolds number within a regime.
* The slipstream (blown-wing) model only credits the extra lift from the fan's slipstream
  (:math:`\Delta C_l`); the associated drag and pitching-moment increments are left at zero
  (:math:`\Delta C_d = \Delta C_m = 0`). This is a deliberate first pass, meant to be replaced by
  a proper OpenVSP-based surrogate later, mirroring the propeller's more complete slipstream
  chain.
* The slipstream model does not propagate the downstream streamtube contraction (contraction
  ratio implicitly 1) or use a spanwise-local wing lift coefficient (it uses the wing's overall
  :math:`C_{l,clean}` instead) -- both simplifications relative to the propeller's slipstream
  chain, reasonable for a fan installed close to the wing rather than far ahead of it (e.g. a nose
  propeller).

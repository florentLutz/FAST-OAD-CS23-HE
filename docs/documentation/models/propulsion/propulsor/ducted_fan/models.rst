.. _models-ducted-fan:

=======================
Ducted fan computation
=======================

.. contents::

*******************
Sizing computation
*******************

Rotor and duct mass
====================
The mass of a single ducted fan unit (rotor + duct, motor and ESC excluded -- those are sized by
their own dedicated components elsewhere in the power train) is estimated from an empirical fit to
two commercial ducted fan units:

.. math::

    m_{rotor} = K_{rotor} \cdot n_{blades} \cdot D^{2.78}

    m_{duct} = K_{duct} \cdot \pi \cdot D \cdot (0.40 \cdot D)

    m = m_{rotor} + m_{duct}

Where :math:`D` is the fan diameter, :math:`n_{blades}` the number of rotor blades, and
:math:`K_{rotor} = 5.02` and :math:`K_{duct} = 8.37` are the calibration constants (the duct chord
is assumed to be 40% of the fan diameter).

Center of gravity
==================
The ducted fan's longitudinal (:math:`x`) position is:

.. math::

    x_{CG} = x_{25\%MAC} - 0.25 \cdot MAC - d_{LE} - 0.5 \cdot depth

Where :math:`d_{LE}` is the distance between the fan and the wing leading edge, and :math:`depth`
is the axial length of the duct.

The lateral (:math:`y`) position is computed from a user-specified span ratio:

.. math::

    y_{CG} = \dfrac{b}{2} \cdot y_{ratio}

Where :math:`b` is the wing span.

Reference chord
=================
The wing chord at the fan's spanwise station is interpolated from the wing's own chord
distribution (as computed by the aerodynamic module) and used downstream by the slipstream model
to convert the fan diameter into a blown wing-area fraction (see `Slipstream computation`_ below).

External drag
==============
Unlike the propeller (a bare rotor with no external shroud, whose parasite drag is taken as zero),
the ducted fan's duct/nacelle has a real wetted surface. Its parasite drag coefficient increment
(referenced to the wing area, to match the propeller's convention) is built up with a classic
nacelle drag buildup:

.. math::

    C_{D0} = C_f \cdot FF \cdot IF \cdot \frac{S_{wet}}{S_{ref}}

.. math::

    C_f = \frac{0.455}{\log_{10}(Re)^{2.58}} \qquad Re = \frac{V_{ref} \cdot depth}{\nu(h_{ref})}

.. math::

    FF = 1 + \frac{0.35}{depth / D} \qquad S_{wet} = \pi \cdot D \cdot depth

Where :math:`C_f` is a turbulent flat-plate skin friction coefficient (Prandtl-Schlichting),
:math:`FF` is the nacelle form factor (Raymer), :math:`IF` is a tunable installation interference
factor (default 1.25), and :math:`V_{ref}`/:math:`h_{ref}` are representative (not
per-mission-point) velocity/altitude settings, one pair for the low-speed condition and one for
cruise.

***********************
Performance computation
***********************

RPM solve
==========
Unlike the propeller (whose rpm follows a prescribed mission schedule), the ducted fan's rpm is
solved implicitly at every mission point so that the fan produces exactly the thrust required by
the aircraft equilibrium:

.. math::

    C_T(J, M_{tip}, \sigma, c/b, \beta) \cdot \rho \cdot \left(\frac{rpm}{60}\right)^2 \cdot D^4
    - T = 0

The thrust coefficient :math:`C_T` (and, from the converged rpm, the power coefficient
:math:`C_P`) are predicted by a surrogate model trained on ducted fan performance data: an SMT
Kriging (KRG) surrogate provides the predicted values, and a PyTorch multilayer perceptron (MLP)
trained on the same data provides exact gradients via automatic differentiation, avoiding the
cost/noise of finite-differencing the Kriging surrogate.

The advance ratio :math:`J` and tip Mach number :math:`M_{tip}` are computed with the classic
ducted fan oblique-inflow correction of Gentry et al. (1998), which accounts for the local flow
incidence angle :math:`\alpha` seen by the fan disk through a duct rectification factor
:math:`k_{duct}`:

.. math::

    V_{eff} = V_\infty \sqrt{\cos^2\alpha + k_{duct}^2 \sin^2\alpha}

    J = \frac{V_{eff}}{(rpm/60) \cdot D} \qquad M_{tip} = \frac{\sqrt{V_{eff}^2 + V_{tip}^2}}{a}

Where :math:`V_{tip} = \pi \cdot (rpm/60) \cdot D` and :math:`a` is the local speed of sound.

Shaft power, torque and maximum values
========================================
Shaft power and torque are derived from the converged rpm and power coefficient, following the
same convention as the propeller. The maximum tip Mach number, advance ratio, rpm and torque
reached over the mission are also reported, for use as sizing constraints.

***********************
Slipstream computation
***********************

The ducted fan is typically installed as part of a distributed electric propulsion, upper-surface
blowing architecture, where the main expected aerodynamic benefit is the extra wing lift generated
by the fan's slipstream blowing over the wing. This is modeled with a deliberately simple,
first-pass approximation reusing two formulas from the propeller's slipstream model
:cite:`de:2019`:

.. math::

    T_c = \frac{T}{\rho V_0^2 D^2} \qquad
    a_p = \frac{1}{2}\left(\sqrt{1 + \frac{8}{\pi}T_c} - 1\right)

.. math::

    \Delta C_l = C_{l,clean} \cdot a_p \cdot \frac{D \cdot c_{ref}}{S_{wing}}

Where :math:`T_c` is the thrust loading, :math:`a_p` the actuator-disk axial induction factor at
the fan disk, :math:`C_{l,clean}` the unblown wing lift coefficient, and :math:`c_{ref}` the wing
chord at the fan's spanwise station (see `Reference chord`_ above). This is the exact reduction of
the full :cite:`de:2019` formula for a fan installed with zero incidence and full immersion in the
wing chord (installation angle :math:`i_p = 0`, height-impact factor :math:`\beta = 1`), reasonable
for a wing-mounted axial fan. The increase in drag and pitching moment caused by the slipstream is
not yet modeled (:math:`\Delta C_d = \Delta C_m = 0`) -- see :ref:`assumptions
<assumptions-ducted-fan>`.

*******************************
Component Computation Structure
*******************************
The following two links are the N2 diagrams representing the performance and sizing computation for
the ducted fan model.

.. raw:: html

   <a href="../../../../../n2_performance_ducted_fan.html" target="_blank">Ducted fan performance N2 diagram</a><br>
   <a href="../../../../../n2_sizing_ducted_fan.html" target="_blank">Ducted fan sizing N2 diagram</a>

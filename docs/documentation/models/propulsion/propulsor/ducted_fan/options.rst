.. _options-ducted-fan:

=========================
Ducted fan model options
=========================

********************
Position option
********************
The ``position`` option only accepts ``on_the_wing``: the ducted fan is installed along the wing
leading edge and blows over the wing (see the :ref:`slipstream computation
<models-ducted-fan>`).

*************************
Duct rectification factor
*************************
``k_duct`` (a ``settings:`` input, default 0.4) is the oblique-inflow duct rectification factor
used in the RPM/advance-ratio computation (0 = perfectly rectified duct flow, 1 = equivalent to an
open propeller). See :ref:`models <models-ducted-fan>` for how it enters the advance ratio and tip
Mach number.

*********************************
Installation interference factor
*********************************
``interference_factor`` (a ``settings:`` input, default 1.25) scales the duct's external drag
buildup to account for installation effects (e.g. wing-duct junction flow) -- typical for a
wing-mounted nacelle/pod.

**********************
Surrogate model paths
**********************
``surrogate_pkl`` and ``grad_surrogate_pkl`` point to the SMT Kriging and PyTorch MLP surrogate
files respectively. They default to the ``.pkl`` files shipped alongside the component and
normally do not need to be overridden.

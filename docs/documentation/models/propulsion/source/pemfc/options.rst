.. _options-pemfc:

=========================================
Proton-exchange membrane fuel cell option
=========================================

*************************
Fuel cell position option
*************************
The Proton-Exchange Membrane Fuel Cell (PEMFC) model has four possible installation position shown as:

| "wing_pod" : Tank installed under the wing with user specified position.
| "underbelly" : Tank installed under and outside the fuselage.
| "in_the_front" : Tank installed at the front section inside the fuselage.
| "in_the_back" : Tank installed at the rear section inside the fuselage.

*******************************
Full system / Pure stack option
*******************************
This option offers flexibility in customizing the Balance of Plants (BoPs) for the PEMFC powertrain. It is controlled by
two adjustment factors, :math:`DAF` and :math:`WAF`, as explained in :ref:`computation logic<models-pemfc>`. To activate this
option, define the expected maximum power density and the specific power surrogate model in the configuration file. By
default, the calculation is set to `fastga_he.submodel.propulsion.performances.pemfc.modeling_option.system`, which
computes the two power-related sizing indicators for full system PEMFC.

.. code-block:: yaml

    submodels:
      ⋮
      submodel.propulsion.performances.pemfc.modeling_option: ...



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

******************************
Maximum current density option
******************************
This option offers flexibility in setting the maximum current density of PEMFC from the powertrain(PT) file shown below.
The maximum current density is an essential parameter that affects the size of the effective area and the layer voltage
of PEMFC. The default value is set to 0.8 :math:`A/cm^2`, which is the reference value taken from :cite:`hoogendoorn:2018`.

.. code-block:: yaml

    power_train_components:
      ⋮
      pemfc_stack_1:
        id: fastga_he.pt_component.pemfc_stack
        position: ...
        options:
          max_current_density: ... # [A/cm^2]



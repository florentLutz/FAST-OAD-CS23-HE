.. _options-pemfc:

=========================================
Proton-exchange membrane fuel cell option
=========================================

*************************
Fuel cell position option
*************************
The PEMFC stack model has four possible installation positions shown as:

| ``wing_pod`` : Stack installed under the wing with user specified position.
| ``underbelly`` : Stack installed under and outside the fuselage.
| ``in_the_front`` : Stack installed at the front section inside the fuselage.
| ``in_the_back`` : Stack installed at the rear section inside the fuselage.

*********************
Model fidelity option
*********************
This option allows to choose between the two polarization models for PEMFC stacks, empirical model and analytical
model. :ref:`The empirical model <models-pemfc-empirical>` is based on the empirical data of Aerostak 200W from
:cite:`hoogendoorn:2018`. :ref:`The analytical model <models-pemfc-analytical>` is derived from standard fuel cell
polarization curve calculation from :cite:`juschus:2021`.


.. code-block:: yaml

    power_train_components:
      ⋮
      pemfc_stack_1:
        id: fastga_he.pt_component.pemfc_stack
        position: ...
        options:
          model_fidelity: ... # "empirical" or "analytical"



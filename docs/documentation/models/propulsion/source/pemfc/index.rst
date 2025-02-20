========================================
Proton-exchange membrane fuel cell model
========================================

This power source model in FAST-OAD-CS23-HE represent PEMFC stack that generates electric energy through the Hydrogen
Oxidation Reaction (HOR) at the anode, where hydrogen gas (:math:`\text{H}_2`) splits into protons (:math:`\text{H}^+`)
and electrons (:math:`\text{e}^-`). The :math:`\text{H}^+` ions pass through the membrane, while the :math:`\text{e}^-`
travel through an external circuit, creating an electric current. At the cathode, the Oxygen Reduction Reaction (ORR)
combines oxygen (:math:`\text{O}_2`) with :math:`\text{H}^+` and :math:`\text{e}^-` to form
water (:math:`\text{H}_2\text{O}`). This process produces electricity with :math:`\text{H}_2\text{O}` as the only byproduct.

.. math::

   \text{Anode (HOR):} \quad \text{H}_2 \rightarrow 2\text{H}^+ + 2e^-

.. math::

   \text{Cathode (ORR):} \quad \frac{1}{2} \text{O}_2 + 2\text{H}^+ + 2e^- \rightarrow \text{H}_2\text{O}

.. math::

   \text{Overall Reaction:} \quad \text{H}_2 + \frac{1}{2} \text{O}_2 \rightarrow \text{H}_2\text{O}


This component can be activated through the powertrain configuration file (PT file). The registered installation
positions can be found at tank position options in :ref:`options <options-pemfc>`.

.. code-block:: yaml

    power_train_components:
      ⋮
      pemfc_stack_1:
        id: fastga_he.pt_component.pemfc_stack
        position: ...
        options: ...

A brief description of the PEMFC component is presented here:

.. _table:
.. toctree::
   :maxdepth: 2

    PEMFC computation logic <models>
    PEMFC options <options>
    PEMFC model assumptions <assumptions>
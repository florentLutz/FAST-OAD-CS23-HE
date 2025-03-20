==========================
Hydrogen fuel system model
==========================

The hydrogen fuel system is a connector option in FAST-OAD-CS23-HE, designed to connect hydrogen storage devices and
hydrogen_powered sources. This component can be activated through the powertrain configuration file (PT file). The
registered installation positions can be found in :ref:`options <options-h2-fuel-system>`. This model is extremely
simplified for temporary usage, the complete model and documentation will be updated in future version.

.. code-block:: yaml

    power_train_components:

      h2_fuel_system_1:
        id: fastga_he.pt_component.h2_fuel_system
        position: ...

A brief description of the hydrogen fuel system component is presented here:

.. _table:
.. toctree::
   :maxdepth: 2

    Hydrogen fuel system computation logic <models>
    Hydrogen fuel system customization options <options>
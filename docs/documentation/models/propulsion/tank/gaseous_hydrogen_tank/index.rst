.. _gh2-tank:

============================
Gaseous hydrogen tank model
============================

The gaseous hydrogen tank is a storage option in FAST-OAD-CS23-HE, designed to store hydrogen in gaseous state.
This component can be activated through the powertrain configuration file (PT file). The registered installation
positions can be found at tank position options in :ref:`options <options-gaseous-hydrogen-tank>`.

.. code-block:: yaml

    power_train_components:

      gaseous_hydrogen_tank_1:
        id: fastga_he.pt_component.gaseous_hydrogen_tank
        position: ...

A brief description of the gaseous hydrogen tank component is presented here:

.. _table:
.. toctree::
   :maxdepth: 2

    Gaseous hydrogen tank computation logic <models>
    Gaseous hydrogen tank customization options <options>
    Gaseous hydrogen tank model assumption <assumptions>
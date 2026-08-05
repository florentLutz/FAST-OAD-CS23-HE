.. _ducted-fan:

=================
Ducted fan model
=================

The ducted fan is a propulsor option in FAST-OAD-CS23-HE: a small-diameter, shrouded rotor
(electric ducted fan, EDF) typically mounted on the wing leading edge as part of a distributed
electric propulsion (DEP) architecture. This component can be activated through the powertrain
configuration file (PT file). The registered installation positions can be found at ducted fan
position options in :ref:`options <options-ducted-fan>`.

.. code-block:: yaml

    power_train_components:

      ducted_fan_1:
        id: fastga_he.pt_component.ducted_fan
        position: on_the_wing

A brief description of the ducted fan component is presented here:

.. _table:
.. toctree::
   :maxdepth: 2

    Ducted fan computation logic <models>
    Ducted fan customization options <options>
    Ducted fan model assumptions <assumptions>

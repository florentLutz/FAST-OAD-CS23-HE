.. _options-h2-fuel-system:

==========================
Hydrogen fuel system model
==========================

**********************
System position option
**********************
The hydrogen fuel system is typically placed inside the fuselage, with the power source and storage tank in separate
positions. However, components could also be installed together or integrated along the wing. To support these
configurations, the ``compat`` and ``wing_related`` options are also introduced.

Compact option
**************
The compact configuration option sets the hydrogen fuel system length as the wing MAC, ensuring accurate weight
estimation when the power source and tank are installed at the same position.

Wing-related option
*******************
The wing-related configuration option adjusts the center of gravity and system length when the pipe network extends into
the wing to connect the power source or storage tank.

Fuselage position option
*******************************
The fuselage part of hydrogen fuel system model has three possible installation options:

| "in_the_rear" : Located in the rear of the cabin.
| "in_the_middle" : Located in the middle of the cabin.
| "in_the_front" : Located in the front of the cabin.

All the position-related options can be activated in the PT file:

.. code-block:: yaml

    power_train_components:
      ⋮
      h2_fuel_system_1:
        id: fastga_he.pt_component.h2_fuel_system_1
        position: ...
        options:
          compact:... #True, False
          wing_related:... #True, False
.. _attributes:

=========================================
PT file visualization function attributes
=========================================

**********************
System position option
**********************
The hydrogen fuel system is typically placed inside the fuselage, with the power source and storage tank in separate
positions. However, components could also be installed together or integrated along the wing. To support these
configurations, the ``compact`` and ``wing_related`` options are also introduced.

Compact option
**************
The compact configuration option sets the hydrogen fuel system length as the wing MAC, ensuring accurate weight
estimation when the power source and tank are installed at the same position.

Wing-related option
*******************
The wing-related configuration option adjusts the center of gravity and system length when the pipe network extends into
the wing to connect the power source or storage tank.


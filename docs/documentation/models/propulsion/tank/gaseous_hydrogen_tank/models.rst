.. _models-gaseous_hydrogen_tank:

=================================
Gaseous Hydrogen Tank computation
=================================

.. contents::

*************************
Tank capacity calculation
*************************
The tank capacity is calculated based on the hydrogen capacity in mass specified by user (:math:`M_H`)
and the ideal gas assumption. Thus, the ideal gas constant (:math:`R`), the storage temperature (:math:`T`),
and the storage pressure (:math:`P`)  are applied in this calculation.

.. math::

    V_{inner} = \frac{Z*M_H*R*T}{P}


The Hydrogen gas compressibility factor (:math:`Z`) is expressed as:

.. math::
    Z = 0.99704 + 6.4149*10^{-9}*P


*************************
Tank geometry calculation
*************************

Tank diameter calculation
=========================


The diameter calculation is based on the hoop stress of a cylindrical tank calculation provided by :cite:`colozza:2002`

.. math::

   t_{wall} = \frac {R_{in} * SF*P}{\sigma_{wall}}

As the tank outer diameter (:math:`D_{outer}`) is defined by user, the tank inner diameter (:math:`D_{inner}`) is derived
with the following equation:

.. math::

    D_{inner} = \frac{\sigma_{wall} * D_{outer}}{\sigma_{wall}+ SF*P}

Where :math:`SF` represent the safety factor of the tank,  :math:`P` is the tank storage pressure, and :math:`\sigma_{wall}` is the tank wall material yield stress.


Tank length calculation
=======================
With the assumption given that the shape of the tank is cylindrical with hemispherical cap at both coth end,
the length of the tank can be expressed as:

.. math::

    L = \frac {V_{inner} - V_{cap}} {A_{cross}} + D_{outer}

Where :math:`V_{inner}` denotes the inner volume, as calculated in the tank capacity section.
:math:`V_{cap}` represents the inner volume of the two hemispherical caps,
while :math:`A_{cross}` refers to the tank's inner cross-sectional area.

.. math::

    V_{cap} = \frac{\pi D_{inner}^3}{6} \\
    A_{cross} = \frac{\pi D_{inner}^2}{4}

*******************************
Component Computation Structure
*******************************
The following two links are the N2 diagrams representing the performance and sizing computation
in gaseous hydrogen tank component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_gh2_tank.html" target="_blank">Gaseous hydrogen tank performance N2 diagram</a><br>
   <a href="../../../../../../../n2/n2_sizing_gh2_tank.html" target="_blank">Gaseous hydrogen tank sizing N2 diagram</a>






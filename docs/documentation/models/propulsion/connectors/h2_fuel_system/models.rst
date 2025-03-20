.. _models-hydrogen-fuel-system:

================================
Hydrogen fuel system computation
================================

.. contents::

*************************
Pipe geometry calculation
*************************

System length calculation
=========================

The overall length of the hydrogen fuel system is calculated with summing the multiplications of four different
pipe lengths based on installation positions with the amount that are considered for each type of length respectively.

.. math::

    L_{system} = \sum_{i=\text{front, rear, wing, near}} L_{i} \cdot N_{i}

With

.. math::
    L_{\text{front}} = L_{\text{rear}} = 0.5*L{\text{cabin}} \\
    L_{\text{near}} = MAC_{\text{wing}} \\
    L_{\text{wing}} = 0.5 * span_{\text{wing}} *\lambda{wing}

Where :math:`L{\text{cabin}}` is the cabin length and :math:`\lambda{wing}` is the position in portion of half wing span
that the source is fixed with respect to the wing root.


Pipe diameter calculation
=========================

The inner pipe diameter calculation is based on the hoop stress of a cylindrical tank calculation provided by :cite:`colozza:2002`

.. math::

   t_{wall} = \frac {R_{in} * SF*P}{\sigma_{wall}}

With the pipe outer diameter provided by user, the pipe inner diameter (:math:`D_{inner}`) is derived
with the following equation:

.. math::

    D_{inner} = \frac{\sigma_{wall} * D_{outer}}{\sigma_{wall}+ SF*P}

Where :math:`SF` represent the safety factor of the pipes,  :math:`P` is the pipe pressure, and :math:`\sigma_{wall}` is
the tank wall material yield stress.


*************************
Pressure Loss calculation
*************************



Tank length calculation
=======================
With the assumption that the shape of the tank is cylindrical with hemispherical cap at both end,
the length of the tank can be expressed as:

.. math::

    L = \frac {V_{inner} - V_{cap}} {A_{cross}} + D_{outer}

Where :math:`V_{inner}` denotes the inner volume, as calculated in the tank capacity section,
:math:`V_{cap}` represents the inner volume of the two hemispherical caps,
while :math:`A_{cross}` refers to the tank's inner cross-sectional area.

.. math::

    V_{cap} = \frac{\pi D_{inner}^3}{6} \\
    A_{cross} = \frac{\pi D_{inner}^2}{4}

*******************************
Component Computation Structure
*******************************
The following two links are the N2 diagrams representing the performance and sizing computation
in hydrogen fuel system component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_gh2_tank.html" target="_blank">Gaseous hydrogen tank performance N2 diagram</a><br>
   <a href="../../../../../../../n2/n2_sizing_gh2_tank.html" target="_blank">Gaseous hydrogen tank sizing N2 diagram</a>
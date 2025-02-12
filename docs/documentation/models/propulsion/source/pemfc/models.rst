.. _models-pemfc:

==============================================
Proton-exchange membrane fuel cell computation
==============================================

.. contents::

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



*******************************
Component Computation Structure
*******************************
The following two links are the N2 diagrams representing the performance and sizing computation
in Proton-Exchange Membrane Fuel Cell (PEMFC) component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_pemfc.html" target="_blank">PEMFC performance N2 diagram</a><br>
   <a href="../../../../../../../n2/n2_sizing_pemfc.html" target="_blank">PEMFC sizing N2 diagram</a>






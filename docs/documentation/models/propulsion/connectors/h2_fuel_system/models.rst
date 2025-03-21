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

The overall length of the hydrogen fuel system is calculated by summing the pipe lengths in each configuration and
weighting each configuration with their respective quantities.

.. math::

    L_{system} = \sum_{i=\text{front, rear, wing, near}} L_{i} \cdot N_{i}

With

.. math::
    L_{\text{front}} = L_{\text{rear}} = 0.5 \cdot L_{\text{cabin}} \\
    L_{\text{near}} = MAC_{\text{wing}} \\
    L_{\text{wing}} = 0.5 \cdot S_{\text{wing}} \cdot \lambda{\text{wing}}

Where :math:`L_{\text{cabin}}` is the cabin length, :math:`S_{\text{wing}}` is the wing span,  and :math:`\lambda{wing}`
is the position in portion of half wing span that the source is fixed with respect to the wing root.


Pipe diameter calculation
=========================
The pipe inner diameter is computed based on the hoop stress calculation similar as :ref:`gaseous hydrogen tank inner diameter <models-gaseous_hydrogen_tank-inner-diameter>`.
But different from the one performed in gaseous hydrogen tank model, the pipe diameter :math:`D_{\text{pipe}}`, served
as the outer diameter of the pipe line without insulation, is directly defined by user.



*******************************
Component Computation Structure
*******************************
The following two links are the N2 diagrams representing the performance and sizing computation
in hydrogen fuel system component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_h2_fuel_system.html" target="_blank">Hydrogen fuel system performance N2 diagram</a><br>
   <a href="../../../../../../../n2/n2_sizing_h2_fuel_system.html" target="_blank">Hydrogen fuel system sizing N2 diagram</a>
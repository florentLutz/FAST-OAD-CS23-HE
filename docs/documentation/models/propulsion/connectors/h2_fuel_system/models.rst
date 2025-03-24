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

The hydrogen fuel system length depends on three three installation types for the power source and storage tank. In the
compact configuration,the components are installed in the same position. The wing-related configuration places one
component along the wing. The ordinary configuration installs the components separately within the fuselage. With these
three configurations, the following equations calculate the system length for each setup.

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
The pipe inner diameter is computed based on the hoop stress calculation similarly to what is done in :ref:`gaseous hydrogen tank inner diameter <models-gaseous_hydrogen_tank-inner-diameter>`.
However, unlike for the  gaseous hydrogen tank model, the pipe diameter :math:`D_{\text{pipe}}`, is directly defined by user.



*******************************
Component Computation Structure
*******************************
The following two links directs to the N2 diagrams representing the performance and sizing computation
in hydrogen fuel system component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_h2_fuel_system.html" target="_blank">Hydrogen fuel system performance N2 diagram</a><br>
   <a href="../../../../../../../n2/n2_sizing_h2_fuel_system.html" target="_blank">Hydrogen fuel system sizing N2 diagram</a>
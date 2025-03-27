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

The hydrogen fuel system length depends on three installation types for the power source and storage tank. In the
compact configuration, the components are installed in the same location. The wing-related configuration places at least one
component along the wing. The default configuration installs the components separately within the fuselage. With these
three configurations, the following equations calculate the system length for each setup.

.. math::

    L_{system} =
    \begin{cases}
        MAC_{\text{wing}} & \text{if compact} \\
        L_{fus} + L_{\text{wing}} & \text{else}
    \end{cases}

Where,

.. math::

    L_{\text{wing}} =
    \begin{cases}
        0.5 * b_{\text{wing}} * \lambda_{\text{wing}} & \text{if wing-related} \\
        0.0 & \text{else}
    \end{cases}

.. math::

    L_{fus} =
    \begin{cases}
        L_{\text{cabin}} & \text{if in the middle} \\
        0.5 * L_{\text{cabin}} & \text{else}
    \end{cases}

Where :math:`L_{\text{cabin}}` is the cabin length, :math:`b_{\text{wing}}` is the wing span,  and :math:`\lambda{\text{wing}}`
is the position, as a ratio of half wing span, where the source is located. The position options of pipe network length
along fuselage :math:`L_{fus}` and length along wing :math:`L_{\text{wing}}` can be found in :ref:`options <options-h2-fuel-system>`.


Pipe diameter calculation
=========================
The pipe inner diameter is computed based on the hoop stress calculation similarly to what is done in :ref:`gaseous hydrogen tank inner diameter <models-gaseous_hydrogen_tank-inner-diameter>`.
However, unlike for the  gaseous hydrogen tank model, the pipe diameter :math:`D_{\text{pipe}}`, is directly defined by user.

.. image:: ../../../../../img/h2_pipe.svg
    :width: 600
    :align: center

This figure illustrates the main geometrical parameters of the pipe cross section.

*******************************
Component Computation Structure
*******************************
The following two links directs to the N2 diagrams representing the performance and sizing computation
for the hydrogen fuel system component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_h2_fuel_system.html" target="_blank">Hydrogen fuel system performance N2 diagram</a><br>
   <a href="../../../../../../../n2/n2_sizing_h2_fuel_system.html" target="_blank">Hydrogen fuel system sizing N2 diagram</a>
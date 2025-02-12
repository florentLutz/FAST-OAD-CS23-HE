.. _models-pemfc:

==============================================
Proton-exchange membrane fuel cell computation
==============================================

.. contents::

***********************************
Fuel cell Layer voltage calculation
***********************************
The Proton-Exchange Membrane Fuel Cell (PEMFC) stack are consisted with multiple layers of single layer PEMFC. With the
maximum current density (:math:`A/cm^2`) derived with the constraint component to ensure positive voltage, the PEMFC
layer voltage can be calculated with these two polarization curve derived from different PEMFC systems.

Simple PEMFC polarization model
===============================

.. math::
    V = V_0 - B \cdot \ln{(j)} - R \cdot j - m \cdot e^{n \cdot j} + C \cdot \ln{(\frac{P_{op}}{P_{amb}})} \\
    C = -0.0032  \ln{(\frac{P_{op}}{P_{amb}})} ^ 2 + 0.0019 \ln{(\frac{P_{op}}{P_{amb}})} + 0.0542

Analytical PEMFC polarization model
===================================

.. math::
    V = VAF \left[ E_0 - \frac{\Delta S}{2Fr}(T - T_0) + \frac{RT}{2Fr} \ln \left( p_{H_2} \sqrt{p_{O_2}} \right) -
    \frac{RT}{\alpha Fr} \ln \left( \frac{j + j_{leak}}{j_0} \right) - rj
    - c \ln \left( \frac{j_{lim}}{j_{lim} - j - j_{leak}} \right) \right] \\
    VAF = -0.022830 P_{\text{op}}^4 + 0.230982 P_{\text{op}}^3
        - 0.829603 P_{\text{op}}^2 + 1.291515 P_{\text{op}} + 0.329935


******************************
Sizing calculation
******************************

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






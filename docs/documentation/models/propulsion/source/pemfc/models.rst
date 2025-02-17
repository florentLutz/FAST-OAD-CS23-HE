.. _models-pemfc:

==============================================
Proton-exchange membrane fuel cell computation
==============================================

.. contents::

***********************************
Fuel cell Layer voltage calculation
***********************************
The Proton-Exchange Membrane Fuel Cell (PEMFC) stack is consisted with multiple layers of single layer PEMFC. With the
maximum current density (:math:`A/cm^2`) derived with the constraint component to ensure positive voltage, the PEMFC
layer voltage can be calculated with these two polarization curve derived from different PEMFC systems. The general
expression of the fuel cell polarization model is expressed as :eq:`_general_pemfc_iv`

.. _general_pemfc_iv:
.. math::

   V_{\text{operating}} = V_r - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}}

The :math:`V_{\text{operating}}` represents the operating voltage of the fuel cell under standard conditions, while
:math:`V_r` is the reversible open-circuit voltage, determined by the Gibbs free energy of the chemical reaction.
:math:`V_{\text{activation}}` corresponds to the activation loss caused by the overpotential at both electrodes,
particularly the oxygen reduction reaction overpotential. :math:`V_{\text{ohmic}}` denotes the ohmic loss due to the
electrical resistance of the electrodes, and :math:`V_{\text{mass-transport}}` represents the mass-transport loss, which
occurs when reactant gases, such as oxygen or fuel, face diffusion limitations at the electrodes.

Simple PEMFC polarization model
===============================
The simple PEMFC polarization model is based on an empircal model of Aerostak 200W PEMFC derived by
:cite:`hoogendoorn:2018`. This model utilize the empirical open circuit voltage :math:`V_0` and the voltage losses in
simplified form obtained with curve fitting. The voltage difference due to operating pressure variation is also
considered in this model shown as :math:`\Delta V_p`.

.. math::
    V = V_0 - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}} + \Delta V_p \\

With

.. math::
    V_{\text{activation}} = B \cdot \ln{(j)} \\
    V_{\text{ohmic}} =  R \cdot j \\
    V_{\text{mass-transport}} =  m \cdot e^{n \cdot j} \\
    \Delta V_p = C \cdot \ln{(\frac{P_{op}}{P_{nom}})} \\
    C = -0.0032  \ln{(\frac{P_{op}}{P_{nom}})} ^ 2 + 0.0019 \ln{(\frac{P_{op}}{P_{nom}})} + 0.0542

And

.. raw:: html

   <div align="center">

+------------+-----------+----------------+
| Parameter  | Value     | Unit           |
+------------+-----------+----------------+
| V₀         | 0.83      | [V]            |
+------------+-----------+----------------+
| B          | 0.014     | [V/ln(A/cm²)]  |
+------------+-----------+----------------+
| R          | 0.24      | [Ω cm²]        |
+------------+-----------+----------------+
| m          | 5.63E-06  | [V]            |
+------------+-----------+----------------+
| n          | 11.42     | [cm²/A]        |
+------------+-----------+----------------+

.. raw:: html

   </div>

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






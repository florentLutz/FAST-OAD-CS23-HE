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
expression of the fuel cell polarization model is expressed as below given by :cite:`dicks:2018`.

.. math::

   V_{\text{operating}} = V_r - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}}

The :math:`V_{\text{operating}}` represents the operating voltage of the fuel cell under standard conditions, while
:math:`V_r` is the reversible open-circuit voltage, determined by the Gibbs free energy of the chemical reaction.
:math:`V_{\text{activation}}` corresponds to the activation loss caused by the kinetic energy barrier at both electrodes
with low current density :cite:`juschus:2021`, particularly the oxygen reduction reaction. :math:`V_{\text{ohmic}}`
denotes the ohmic loss due to the electrical resistance of the electrodes, and :math:`V_{\text{mass-transport}}`
represents the mass-transport loss, which occurs when reactant gases, such as oxygen or fuel, face diffusion limitations
at the electrodes.

Simple PEMFC polarization model
===============================
The simple PEMFC polarization model is based on an empircal model of Aerostak 200W PEMFC derived by
:cite:`hoogendoorn:2018`. This model utilize the empirical open circuit voltage :math:`V_0` and the voltage losses in
simplified form obtained with curve fitting. The voltage deviation due to operating pressure variation is also
considered in this model shown as :math:`\Delta V_p`. The pressure ratio :math:`P_R` is the ration between the operating pressure :math:`P_{op}` and the nominal operating
pressure :math:`P_{nom}`. The unit of current density :math:`j` is expressed in [A/cm²] for this model.

.. math::
    V = V_0 - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}} + \Delta V_p \\

With

.. math::
    V_{\text{activation}} = B \cdot \ln{(j)} \\
    V_{\text{ohmic}} =  R \cdot j \\
    V_{\text{mass-transport}} =  m \cdot e^{n \cdot j} \\
    P_R = \frac{P_{op}}{P_{nom}} \\
    \Delta V_p = C \cdot \ln{(P_R)} \\
    C = -0.0032  \ln{(P_R)} ^ 2 + 0.0019 \ln{(P_R)} + 0.0542

And

.. raw:: html

   <div align="center">

=========  =========  ============
Parameter  Value      Unit
=========  =========  ============
V₀         0.83         V
B          0.014       V/ln(A/cm²)
R          0.24        Ω cm²
m          5.63*10⁻⁶   V
n          11.42       cm²/A
Pₙₒₘ        101325      Pa
=========  =========  ============


.. raw:: html

   </div>

Analytical PEMFC polarization model
===================================
The analytical PEMFC polarization model is based on the thermodynamic characteristics of fuel cells, as outlined in
:cite:`juschus:2021`. It accounts for voltage losses under typical operational conditions, as well as variations in
operating temperature and pressure, represented by :math:`V_T` and :math:`V_{P_e}`, respectively. The variable
:math:`p_{O_2}` denotes the operating pressure at the cathode, :math:`p_{H_2}` refers to the operating pressure at the
anode, and :math:`T` is the operating temperature of the fuel cell. The constants :math:`R` and :math:`Fr` represent the
gas constant and Faraday's constant. The voltage adjustment factor :math:`VAF` adjusts for changes in ambient pressure
:math:`P_{\text{amb}}`. The current density, :math:`j`, is expressed in [A/m²] for this model.

.. math::
    V = VAF [E_0 - V_T + V_{P_e} - V_{\text{activation}} - V_{\text{ohmic}}
    - V_{\text{mass-transport}}] \\

With

.. math::
    V_T = \frac{\Delta S}{2Fr}(T - T_0) \\
    V_{P_e} = \frac{RT}{2 Fr} \ln( p_{H_2} \sqrt{p_{O_2}}) \\
    V_{\text{activation}} = \frac{RT}{\alpha Fr} \ln \left( \frac{j + j_{leak}}{j_0} \right) \\
    V_{\text{ohmic}} = r \cdot j \\
    V_{\text{mass-transport}} = c \ln \left( \frac{j_{lim}}{j_{lim} - j - j_{leak}} \right)\\
    VAF = -0.022830 P_{\text{amb}}^4 + 0.230982 P_{\text{amb}}^3
        - 0.829603 P_{\text{amb}}^2 + 1.291515 P_{\text{amb}} + 0.329935

And

.. raw:: html

   <div align="center">

=========  ======  ===========
Parameter  Value   Unit
=========  ======  ===========
E₀         1.229   V
ΔS         44.34   J/(mol·K)
T₀         289.15  K
α           0.3    –
ε           0.5    V
r           10⁻⁶    Ω·m²
jₗᵢₘ        20000   A/m²
jₗₑₐₖ         100    A/m²
j₀          1.0    A/m²
=========  ======  ===========

.. raw:: html

   </div>

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






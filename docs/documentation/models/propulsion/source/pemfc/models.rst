==============================================
Proton-exchange membrane fuel cell computation
==============================================

.. contents::

***********************************
Fuel cell Layer voltage calculation
***********************************
The Proton-Exchange Membrane Fuel Cell (PEMFC) system is composed of multiple single-layer PEMFCs, each
serving as a base unit of the overall stack. The single layer operating voltage is calculated by subtracting the
reversible open-circuit voltage with the losses from different factors. The equation below, from :cite:`dicks:2018`, is
the general representation of the fuel cell polarization model that demonstrates the calculation of single layer operating voltage.

.. math::

   V_{\text{operating}} = V_r - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}}

The :math:`V_{\text{operating}}` represents the operating voltage of the fuel cell under standard conditions, while
:math:`V_r` is the reversible open-circuit voltage, determined by the Gibbs free energy of the chemical reaction.
:math:`V_{\text{activation}}` corresponds to the activation loss caused by the kinetic energy barrier at both electrodes
:cite:`juschus:2021`, :math:`V_{\text{ohmic}}` denotes the ohmic loss due to the electrical resistance of the electrodes
, and :math:`V_{\text{mass-transport}}` represents the mass-transport loss, which occurs when reactant gases, such as
oxygen or fuel, face diffusion limitations at the electrodes.

There are two polarization curve models implemented in this component to model single layer operating voltage. The empirical PEMFC polarization
model is based on Aerostak 200W PEMFC derived by :cite:`hoogendoorn:2018`. The analytical PEMFC polarization model is
based on the thermodynamic characteristics of fuel cells, as outlined in :cite:`juschus:2021`.

.. _models-pemfc-empirical:

Empirical PEMFC polarization model
==================================
This model utilizes the empirical open circuit voltage :math:`V_0` and the voltage losses in
simplified form obtained with curve fitting in :cite:`hoogendoorn:2018`. The voltage deviation due to operating pressure variation is also
considered in this model shown as :math:`\Delta V_p`. The pressure ratio :math:`P_R` is the ratio between the operating
pressure :math:`P_{op}` and the nominal operating pressure :math:`P_{nom}`. The unit of current density :math:`j` is
expressed in [:math:`A/cm^2`] for this model.

.. math::
    V = V_0 - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}} + \Delta V_p \\

With

.. math::
    V_{\text{activation}} = B \cdot \ln{(j)} \\[10pt]
    V_{\text{ohmic}} =  R \cdot j \\[10pt]
    V_{\text{mass-transport}} =  m \cdot e^{n \cdot j} \\[10pt]
    P_R = \frac{P_{op}}{P_{nom}} \\[10pt]
    \Delta V_p = C \cdot \ln{(P_R)} \\[10pt]
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

This table proivdes the parameter values that has been considered to model Aerostak 200W in hoogendoorn's research
:cite:`hoogendoorn:2018`.

.. _models-pemfc-analytical:

Analytical PEMFC polarization model
===================================
This moodel accounts for voltage losses under typical operational conditions, as well as variations in operating
temperature and pressure, represented by :math:`V_T` and :math:`V_{P_e}`, respectively. The variable :math:`p_{O_2}`
denotes the operating pressure at the cathode, :math:`p_{H_2}` refers to the operating pressure at the anode, and
:math:`T` is the operating temperature of the fuel cell. The constants :math:`R` and :math:`Fr` are the gas constant
and Faraday's constant. The pressure voltage correction :math:`\kappa_{vc}`, obtained from
`juschus' github repository <https://github.com/danieljuschus/pemfc-aircraft-sizing>`_ , adjusts for changes in ambient
pressure :math:`P_{\text{amb}}`. The current density, :math:`j`, is expressed in [:math:`A/m^2`] for this model.

.. math::
    V = \kappa_{vc} [E_0 - V_T + V_{P_e} - V_{\text{activation}} - V_{\text{ohmic}} - V_{\text{mass-transport}}]

With

.. math::

    V_T = \frac{\Delta S}{2Fr}(T - T_0) \\[10pt]
    V_{P_e} = \frac{RT}{2 Fr} \ln( p_{H_2} \sqrt{p_{O_2}}) \\[10pt]
    V_{\text{activation}} = \frac{RT}{\alpha Fr} \ln \left( \frac{j + j_{leak}}{j_0} \right) \\[10pt]
    V_{\text{ohmic}} = r \cdot j \\[10pt]
    V_{\text{mass-transport}} = \epsilon \ln \left( \frac{j_{lim}}{j_{lim} - j - j_{leak}} \right) \\[10pt]
    \kappa_{vc} = -0.022830 P_{\text{amb}}^4 + 0.230982 P_{\text{amb}}^3 - 0.829603 P_{\text{amb}}^2 + 1.291515 P_{\text{amb}} + 0.329935


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

This table provides the parameter values that has been considered in juschus' research :cite:`juschus:2021`.

******************
Sizing calculation
******************
PEMFC dimension calculation
===========================
The PEMFC stack length is calculated by multiplying the number of layers, :math:`N_{layers}`, with the cell length.
:math:`L_c` is the cell length calculates from dividing total length of the Aerostak 200W by the number of single-layered
fuel cells.

.. math::
   L_{pemfc} = L_c \cdot N_{layers}

Then, utilizing the PEMFC stack volume calculated with the maximum design power :math:`P_{max}` produced by PEMFC and the
power density of the fuel cell :math:`\rho_{power}`, the cross-section area :math:`A_{cross}` is obtained as:

.. math::
    A_{cross} = \frac { k_{volume} \cdot P_{max}} {\rho_{power}  \cdot L_{pemfc}}

Where the volume tuning factor :math:`k_{volume}` allows users to manually adjust the volume of the PEMFC stack.

Finally, the height :math:`H_{pemfc}` and width :math:`W_{pemfc}` of the PEMFC stack can be obtained as:

.. math::

   H_{pemfc} = \sqrt{0.5 A_{cross}} \\
   W_{pemfc} = \sqrt{2 A_{cross}} \\
    \text{if positioned underbelly}

.. math::
    H_{pemfc} = W_{pemfc} = \sqrt{A_{cross}} \\
    \text{if positioned inside fuselage or wing pod}

PEMFC weight calculation
========================
The PEMFC stack weight is calculated with the cell density :math:`\rho_{cell}` of Aerostak 200W provided by
:cite:`hoogendoorn:2018`, which is the total weight divided by the total effective area of the Aerostak 200W PEMFC stack.
The weight of the PEMFC stack can be expressed as:

.. math::

    M_{pemfc} =k_{mass} \cdot \lambda_{sp} \cdot \rho_{cell} \cdot A_{eff} \cdot N_{layers}

Where :math:`A_{eff}` is the effective area, :math:`N_{layers}` is number of layers, and :math:`\lambda_{sp}` is the
specific power ratio. :math:`\lambda_{sp}` is calculated as the specific power of the Aerostak 200W divided by the
specific power of the PEMFC stack. The mass tuning factor :math:`k_{mass}` allows users to manually adjust
the weight of the PEMFC stack.

*******************************
Component Computation Structure
*******************************
The following three links are the N2 diagrams representing the performance for both polarization models and sizing
computation in Proton-Exchange Membrane Fuel Cell (PEMFC) stack component.

.. raw:: html

   <a href="../../../../../../../n2/n2_performance_pemfc_empirical.html" target="_blank">PEMFC stack performance N2 diagram with empirical polarization model</a><br>
   <a href="../../../../../../../n2/n2_performance_pemfc_analytical.html" target="_blank">PEMFC stack performance N2 diagram with analytical polarization model</a><br>
   <a href="../../../../../../../n2/n2_sizing_pemfc.html" target="_blank">PEMFC stack sizing N2 diagram</a>






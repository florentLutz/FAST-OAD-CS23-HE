.. _models-lca:

======================
Life Cycle Cost models
======================
.. image:: ../../../img/cost_computation_structure.svg
    :width: 800
    :align: center

The Life Cycle Cost (LCC) of the aircraft consists with two main categories the production cost and the operation cost.
The elements of the both groups are detailed in the following descriptions.

.. contents::

*********************
Production cost model
*********************
The production costs in aircraft level and powertrain level are the two main cost groups of the production cost. Each
aircraft level production cost is calculated the Eastlake model (1986) from :cite:`gudmundsson:2013`. The component
purchase cost of the powertrain components are either based on the Eastlake model (1986) from :cite:`gudmundsson:2013`
or estimated from the retailers or the company websites. The cost calculations from Gudmundsson are based on the USD in 2012.
To cosider the inflation cost, the cost adjust factor :math:`CPI_{\text{2012}}` is introduced. This factor represent
the cost price index difference between 2012 and current time.

Aircraft level cost
===================

Cost of engineering, toolong, and manufacturing
***********************************************
The cost of the engineering labor, tooling labor, and manufacturing share a similar computation structure. It begins
with estimating the total man-hours required for the development process and for production during the first five years
after the aircraft's launch. Then, the cost is simply calculated with the multiplication of the human-hour
(:math:`H_{\text{labor}}`) and cost rate of labor (:math:`R_{\text{labor}}`) for each subcategory with the inflation
adjustment.

.. math::

    H_{\text{labor}} = f(W_{\text{airframe}},N,V_H,Q_m,F) \\
    C_{\text{labor}} = 2.0969 \cdot H_{\text{labor}} \cdot R_{\text{labor}} \cdot CPI_{\text{2012}}

:math:`W_{\text{airframe}}` is the weight of airframe, :math:`N` is the number of aircraft predicted for a
five-year-period, :math:`V_H` maximum cruise true airspeed in knots, :math:`Q_m` is the estimated aircraft production
rate per month, and :math:`F`,detailed in :cite:`gudmundsson:2013`,is the combination of factors based on aircraft design
specifications.

Cost of development support
***************************
Indirect development labor cost such as overheads,administration, logistics, human resources, facilities maintenance during
the aircraft development. As numerous types of labor are included, this cost can not be estimated with a fixed cost rate.

.. math::

    C_{\text{dev}} = 0.06458 \cdot W_{\text{airframe}}^{0.873} \cdot V_H^{1.89} \cdot N_p^{0.346} \cdot CPI_{\text{2012}}
                    \cdot F

:math:`N_p` is the number of prototype produced during the development period.

Cost of flight test
*******************
Cost of development and certification flight test.

.. math::

    C_{\text{ft}} = 0.009646 \cdot W_{\text{airframe}}^{1.16} \cdot V_H^{1.3718} \cdot N_p^{1.281} \cdot CPI_{\text{2012}}


Cost of quality control
***********************
Cost of manufacturing quality control, which consist with the cost of technicians and equipments.

.. math::

    C_{\text{QC}} = 0.13 * C_{\text{MFG}} * (1 + 0.5f_{\text{comp}})

:math:`C_{\text{MFG}}` is the manufacturing cost of a five-year period, :math:`f_{\text{comp}}` represents the
proportion of the airframe made of composite material.

Cost of material
****************

.. math::

    C_{\text{material}} = 24.896 \cdot  W_{\text{airframe}}^{0.689} \cdot V_H^{0.624} \cdot N^{0.792}
                            \cdot CPI_{\text{2012}} \cdot F_{CF} \cdot F_{\text{press}}

:math:`F_{CF}` is the complex flap system factor = 1.02 if comp[lex flap system applied, :math:`F_{\text{press}}`
is the pressurized factor = 1.01 if pressurized.

Cost of certify
***************

The cost of certification is the sum of engineering labor cost, development support cost, flight test cost, and the
tooling labor cost.

.. math::

    C_{\text{certify}} =  C_{\text{eng}} + C_{\text{dev}} + C_{\text{ft}} + C_{\text{tool}}


Powertrain level cost
=====================

Engine purchase cost
********************
Unit purchase cost of the engine from :cite:`gudmundsson:2013`.

.. math::

    C_{\text{engine}} =
    \begin{cases}
        174 \cdot CPI_{\text{2012}} \cdot P_{\text{BHP}} & \text{if ICE} \\
        377.4 \cdot CPI_{\text{2012}} \cdot P_{\text{SHP}} & \text{if turboshaft}
    \end{cases}

:math:`P_{\text{BHP}}` is the brake-horse power of the internal combustion engine and :math:`P_{\text{SHP}}` of the
turboshaft engine.

Propeller purchase cost
***********************
Unit purchase cost of the propeller from :cite:`gudmundsson:2013`.

.. math::

    C_{\text{propeller}} =
    \begin{cases}
        3145 \cdot CPI_{\text{2012}} & \text{if fixed-pitch} \\
        209.69 \cdot CPI_{\text{2012}} \cdot D_p^2 (\frac {P_{\text{SHP}}}{D_p}) ^{0.12} & \text{if constant-speed}
    \end{cases}

:math:`D_p` is the diameter of the propeller and :math:`P_{\text{SHP}}` is the shaft horse power applied to the propeller.

Synchronous Motor / Generator purchase cost
*******************************************
Unit purchase cost obtained from https://emrax.com/e-motors/.

.. math::

    C_{\text{motor}} = 893.51 \cdot e^{0.0281 P_{\text{max, cont.}} }

:math:`P_{\text{max, cont.}}` is the maximum continuous power of the motor / generator.

Battery purchase cost
*********************
Unit purchase cost obtained with logarithmic regression from :cite:`Wesley:2023`.

.. math::

    C_{\text{bat}} = C_{2022}  E_{\text{bat}} (1.01 - 0.156 \ln{Y_{2022}})

:math:`C_{2022}` is the energy per dollar of battery in 2022, :math:`E_{bat}` is the maximum energy supply from battery,
and :math:`Y_{2022}` is the amount of year from 2022.


********************
Operation cost model
********************
Similar as the production cost model, the operation cost is also built with the cost from aircraft level and powertrain
level. However, to better estimate the regular maintenance cost, the calculation of maintenance is achieved with a
regression model derived with the data from https://www.guardianjet.com/jet-aircraft-online-tools.

Aircraft level cost
===================

Annual loan cost
****************
If the aircraft is fully or partially financed by loaning, the annual payback amount is estimated with the formula based
on regular house mortgage from :cite:`gudmundsson:2013`.

.. math::

   C_{\text{loan}} = \frac{P \cdot R_{\text{interest}}}{1-\frac{1}{(1 + R_{\text{interest}})^n}}

:math:`P` is the principal of the loan, :math:`R_{\text{interest}}` is the annual interest rate, and :math:`n` is the
payback perios.

Powertrain level cost
=====================

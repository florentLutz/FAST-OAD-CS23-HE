.. _models-gaseous_hydrogen_tank:

============================
Gaseous Hydrogen Tank computation
============================

.. contents::

*****************
Key Calculations
*****************

Tank Capacity calculation
====================

.. math::

   t_{wall} = \frac {R_{in} * SF*P}{\sigma_{wall}}

Where $t_{wall}$


Tank Diameter calculation
====================
The diameter calculation is based on the hoop stress of a cylindrical tank calculation provided by :cite:`colozza:2002`

.. math::

   t_{wall} = \frac {R_{in} * SF*P}{\sigma_{wall}}

Where $t_{wall}$ is the thickness of the tank wall,
$R_{in}$ in the inner diameter of the tank,
$SF$ is the safety factor applied to the tank sizing,
$P$ is the storage pressure of the tank,
and $\sigma_{wall}$ is the wall material yield stress.



*****************
Component Computation Structure
*****************

Gaseous Hydrogen Tank performance calculation N2 Diagram
====================
The following N2 diagram demonstrates the connections of each performance computations and the calculation order in this component.

.. raw:: html

    <iframe src="./n2/n2_performance.html" style="border:none; width:100%; height:600px;"></iframe>


Gaseous Hydrogen Tank sizing calculation N2 Diagram
====================
The following N2 diagram demonstrates the connections of each sizing computations and the calculation order in this component.

.. raw:: html

    <iframe src="./n2/n2_sizing.html" style="border:none; width:100%; height:600px;"></iframe>




This is a Work-In-Progress, future version of the documentation will include a description of:

* How the `EcoInvent <https://ecoinvent.org/>`_ database is used
* How the `LCAv package <https://github.com/felixpollet/LCAv>`_ is used
* How the LCA module works in terms of creating an LCA configuration file
* What the functional unit is
* How we compute the impacts per functional unit
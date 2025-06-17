.. _constraint-id:

=====================================
Powertrain Component Constraints & ID
=====================================

**********
Connectors
**********

Harness (DC cable)
==================

DC bus
======

DC splitter
===========

DC-DC converter
===============

Inverter
========

Rectifier
=========

Solid state power controller (SSPC)
===================================

Fuel system
===========

Hydrogen fuel system
====================

Gearbox
=======

Planetary gear
==============

Speed reducer
=============

*****
Loads
*****

DC_loads
========

Permanent magnet synchronous motor (PMSM)
==========================================

*********
Propulsor
*********

Propeller
=========

ID
**

"fastga_he.pt_component.propeller"

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.propeller.torque: fastga_he.submodel.propulsion.constraints.propeller.torque.enforce
    submodel.propulsion.constraints.propeller.rpm: fastga_he.submodel.propulsion.constraints.propeller.rpm.enforce
*******
Sources
*******

Battery
=======

Generator
=========

Internal combustion Engine (ICE)
================================

High RPM ICE
============

Proton-exchange membrane fuel cell (PEMFC)
==========================================

Simple turbo generator
======================

Turboshaft
==========

*****
Tanks
*****

Fuel tank
=========

Gaseous hydrogen tank
=====================

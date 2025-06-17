.. _constraint-id:

=====================================
Powertrain Component Constraints & ID
=====================================

**********
Connectors
**********

Harness (DC cable)
==================

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_line

DC bus
======

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_bus

DC splitter
===========

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_splitter

DC-DC converter
===============

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_dc_converter

Inverter
========

ID
**

.. code:: yaml

    fastga_he.pt_component.inverter

Rectifier
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.rectifier

Solid state power controller (SSPC)
===================================

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_sspc

Fuel system
===========

ID
**

.. code:: yaml

    fastga_he.pt_component.fuel_system

Hydrogen fuel system
====================

ID
**

.. code:: yaml

    fastga_he.pt_component.h2_fuel_system

Gearbox
=======

ID
**

.. code:: yaml

    fastga_he.pt_component.gearbox

Planetary gear
==============

ID
**

.. code:: yaml

    fastga_he.pt_component.planetary_gear

Speed reducer
=============

ID
**

.. code:: yaml

    fastga_he.pt_component.speed_reducer

*****
Loads
*****

DC_loads
========

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_load

Permanent magnet synchronous motor (PMSM)
==========================================

ID
**

.. code:: yaml

    fastga_he.pt_component.pmsm

*********
Propulsor
*********

Propeller
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.propeller

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

ID
**

.. code:: yaml

    fastga_he.pt_component.battery_pack

Generator
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.generator

Internal combustion Engine (ICE)
================================

ID
**

.. code:: yaml

    fastga_he.pt_component.internal_combustion_engine

High RPM ICE
============

ID
**

.. code:: yaml

    fastga_he.pt_component.internal_combustion_engine_high_rpm

Proton-exchange membrane fuel cell (PEMFC)
==========================================

ID
**

.. code:: yaml

    fastga_he.pt_component.pemfc_stack

Simple turbo generator
======================

ID
**

.. code:: yaml

    fastga_he.pt_component.turbo_generator_simple

Turboshaft
==========

ID
**

.. code:: yaml

    fastga_he.pt_component.turboshaft

*****
Tanks
*****

Fuel tank
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.fuel_tank

Gaseous hydrogen tank
=====================

ID
**

.. code:: yaml

    fastga_he.pt_component.gaseous_hydrogen_tank

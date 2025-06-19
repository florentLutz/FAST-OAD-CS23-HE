.. _constraint-id:

=====================================
Powertrain Component Constraints & ID
=====================================
This section lists all component's `id` and their corresponding constraints. Each component `id` is used to configure the
:ref:`PT file <pt-file>`. Constraints for the components defined in the powertrain are specified using service and submodel
naming strings in the `submodels` section of the `configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_.

.. code:: yaml

    submodels:
      service naming string: submodel naming string

- The **service naming string**:  identifies the component and the specific variable being constrained.
- The **submodel naming string**: defines the strictness level of the constraint.

Each component constraint has two levels of strictness:

- **Ensure**: A soft constraint that guides the variable toward a target value.
- **Enforce**: A hard constraint that fixes the variable to a value determined by the model.

To facilitate switching between constraint levels, the submodel naming strings are identical except for the ending:
use ``ensure`` for ensure constraints and ``enforce`` for enforce constraints. By default, constraints are set to
``enforce`` for all components, except for the `voltage` constraint of the PMSM and generator. Since `voltage` is closely
coupled with `RPM` and `torque` constraints, it is recommended to keep it as a soft constraint (``ensure``) to avoid
computational errors.

**********
Connectors
**********

Harness (DC cable)
==================

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_line

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.dc_line.current: fastga_he.submodel.propulsion.constraints.dc_line.current.enforce
    submodel.propulsion.constraints.dc_line.voltage: fastga_he.submodel.propulsion.constraints.dc_line.voltage.enforce

DC bus
======

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_bus

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.dc_bus.current: fastga_he.submodel.propulsion.constraints.dc_bus.current.enforce
    submodel.propulsion.constraints.dc_bus.voltage: fastga_he.submodel.propulsion.constraints.dc_bus.voltage.enforce

DC splitter
===========

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_splitter

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.dc_splitter.current: fastga_he.submodel.propulsion.constraints.dc_splitter.current.enforce
    submodel.propulsion.constraints.dc_splitter.voltage: fastga_he.submodel.propulsion.constraints.dc_splitter.voltage.enforce

DC-DC converter
===============

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_dc_converter

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.dc_dc_converter.current.capacitor: fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.capacitor.enforce
    submodel.propulsion.constraints.dc_dc_converter.current.inductor: fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.inductor.enforce
    submodel.propulsion.constraints.dc_dc_converter.current.module: fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.module.enforce
    submodel.propulsion.constraints.dc_dc_converter.current.input: fastga_he.submodel.propulsion.constraints.dc_dc_converter.current.input.enforce
    submodel.propulsion.constraints.dc_dc_converter.voltage.input: fastga_he.submodel.propulsion.constraints.dc_dc_converter.voltage.input.enforce
    submodel.propulsion.constraints.dc_dc_converter.voltage: fastga_he.submodel.propulsion.constraints.dc_dc_converter.voltage.enforce
    submodel.propulsion.constraints.dc_dc_converter.frequency: fastga_he.submodel.propulsion.constraints.dc_dc_converter.frequency.enforce
    submodel.propulsion.constraints.dc_dc_converter.losses: fastga_he.submodel.propulsion.constraints.dc_dc_converter.losses.enforce
    submodel.propulsion.constraints.dc_dc_converter.input_power: fastga_he.submodel.propulsion.constraints.dc_dc_converter.power.input.enforce

Inverter
========

ID
**

.. code:: yaml

    fastga_he.pt_component.inverter

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.inverter.current: fastga_he.submodel.propulsion.constraints.inverter.current.enforce
    submodel.propulsion.constraints.inverter.voltage: fastga_he.submodel.propulsion.constraints.inverter.voltage.enforce
    submodel.propulsion.constraints.inverter.losses: fastga_he.submodel.propulsion.constraints.inverter.losses.enforce
    submodel.propulsion.constraints.inverter.frequency: fastga_he.submodel.propulsion.constraints.inverter.frequency.enforce
    submodel.propulsion.constraints.inverter.output_power: fastga_he.submodel.propulsion.constraints.inverter.output_power.enforce

Rectifier
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.rectifier

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.rectifier.current.input.rms_one_phase: fastga_he.submodel.propulsion.constraints.rectifier.current.input.rms_one_phase.enforce
    submodel.propulsion.constraints.rectifier.voltage.input.peak: fastga_he.submodel.propulsion.constraints.rectifier.voltage.input.peak.enforce
    submodel.propulsion.constraints.rectifier.losses: fastga_he.submodel.propulsion.constraints.rectifier.frequency.enforce
    submodel.propulsion.constraints.rectifier.frequency: fastga_he.submodel.propulsion.constraints.rectifier.losses.enforce

Solid state power controller (SSPC)
===================================

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_sspc

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.dc_sspc.current: fastga_he.submodel.propulsion.constraints.dc_sspc.current.enforce
    submodel.propulsion.constraints.dc_sspc.voltage: fastga_he.submodel.propulsion.constraints.dc_sspc.voltage.enforce

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

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.gearbox.torque: fastga_he.submodel.propulsion.constraints.gearbox.torque.enforce

Planetary gear
==============

ID
**

.. code:: yaml

    fastga_he.pt_component.planetary_gear

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.planetary_gear.torque: fastga_he.submodel.propulsion.constraints.planetary_gear.torque.enforce

Speed reducer
=============

ID
**

.. code:: yaml

    fastga_he.pt_component.speed_reducer

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.speed_reducer.torque: fastga_he.submodel.propulsion.constraints.speed_reducer.torque.enforce

*****
Loads
*****

DC_loads
========

ID
**

.. code:: yaml

    fastga_he.pt_component.dc_load

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.aux_load.power: fastga_he.submodel.propulsion.constraints.aux_load.power.enforce

Permanent magnet synchronous motor (PMSM)
==========================================

ID
**

.. code:: yaml

    fastga_he.pt_component.pmsm

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.pmsm.torque: fastga_he.submodel.propulsion.constraints.pmsm.torque.enforce
    submodel.propulsion.constraints.pmsm.rpm: fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce
    submodel.propulsion.constraints.pmsm.voltage: fastga_he.submodel.propulsion.constraints.pmsm.voltage.ensure

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

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.battery.state_of_charge: fastga_he.submodel.propulsion.constraints.battery.state_of_charge.enforce

Generator
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.generator

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.generator.torque: fastga_he.submodel.propulsion.constraints.generator.torque.enforce
    submodel.propulsion.constraints.generator.rpm: fastga_he.submodel.propulsion.constraints.generator.rpm.enforce
    submodel.propulsion.constraints.generator.voltage: fastga_he.submodel.propulsion.constraints.generator.voltage.ensure

Internal combustion Engine (ICE)
================================

ID
**

.. code:: yaml

    fastga_he.pt_component.internal_combustion_engine

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.ice.sea_level_power: fastga_he.submodel.propulsion.constraints.ice.sea_level_power.enforce

High RPM ICE
============

ID
**

.. code:: yaml

    fastga_he.pt_component.internal_combustion_engine_high_rpm

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.high_rpm.ice.sea_level_power: fastga_he.submodel.propulsion.constraints.high_rpm_ice.sea_level_power.enforce

Proton-exchange membrane fuel cell (PEMFC)
==========================================

ID
**

.. code:: yaml

    fastga_he.pt_component.pemfc_stack

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.pemfc.effective_area: fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce
    submodel.propulsion.constraints.pemfc.power: fastga_he.submodel.propulsion.constraints.pemfc_stack.power.enforce

Simple turbo generator
======================

ID
**

.. code:: yaml

    fastga_he.pt_component.turbo_generator_simple

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.turbo_generator.power: fastga_he.submodel.propulsion.constraints.turbo_generator.power.enforce

Turboshaft
==========

ID
**

.. code:: yaml

    fastga_he.pt_component.turboshaft

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.turboshaft.rated_power: fastga_he.submodel.propulsion.constraints.turboshaft.rated_power.enforce

*****
Tanks
*****

Fuel tank
=========

ID
**

.. code:: yaml

    fastga_he.pt_component.fuel_tank

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.fuel_tank.capacity: fastga_he.submodel.propulsion.constraints.fuel_tank.capacity.enforce

Gaseous hydrogen tank
=====================

ID
**

.. code:: yaml

    fastga_he.pt_component.gaseous_hydrogen_tank

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.gaseous_hydrogen_tank.capacity: fastga_he.submodel.propulsion.constraints.gaseous_hydrogen_tank.capacity.enforce

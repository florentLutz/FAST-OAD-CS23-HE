.. _constraint-id:

=====================================
Powertrain Component Constraints & ID
=====================================
This section lists all component's `id` and their corresponding constraints. While building a powertrain, any components can be used by adding its `id` to the :ref:`PT file <pt-file>`.. Constraints for
the components defined in the powertrain are specified using service and submodel naming strings in the
`submodels <https://fast-oad.readthedocs.io/en/stable/documentation/custom_modules/add_submodels.html>`_ section of the
`configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_.

.. code:: yaml

    submodels:
      service naming string: submodel naming string

- The **service naming string**:  identifies the component and the specific variable being constrained.
- The **submodel naming string**: defines the strictness level of the constraint.

Each component constraint has two levels of strictness:

- **Ensure**: A soft constraint that keeps the rating condition as an user-defined input, relaxing strict maximum or minimum limits and computing the difference as output.
- **Enforce**: A hard constraint that sets the variable rating based on the maximum or minimum value observed during the mission.

To facilitate switching between constraint levels, the submodel naming strings are identical except for the ending:
use ``ensure`` for ensure constraints and ``enforce`` for enforce constraints. By default, constraints are set to
``enforce`` for all components, except for the `voltage` constraint of the PMSM and generator. Since `voltage` is closely
coupled with `RPM` and `torque` constraints, it is recommended to keep it as a soft constraint (``ensure``) to avoid
computational errors with the current model.

**********
Connectors
**********

DC cable harness
================

The DC cable harness is a connector option in FAST-OAD-CS23-HE, designed to connect DC buses and splitters.
Detailed documentation is still a work in progress.

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

The DC bus is a connector option in FAST-OAD-CS23-HE, designed to connect DC cable harness to an electric component.
Detailed documentation is still a work in progress.

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

The DC splitter is a connector option in FAST-OAD-CS23-HE, designed to connect to multiple electricity sources.
Detailed documentation is still a work in progress.

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

The DC-DC converter is a connector option in FAST-OAD-CS23-HE, designed to rectify voltage for connected electric components.
Detailed documentation is still a work in progress.

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

The inverter is a connector option in FAST-OAD-CS23-HE, designed to convert DC current to AC current.
Detailed documentation is still a work in progress.

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

The rectifier is a connector option in FAST-OAD-CS23-HE, designed to convert AC current to DC current.
Detailed documentation is still a work in progress.

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

The SSPC is a connector option in FAST-OAD-CS23-HE, designed for electric power distribution. It can be opened as an option to disconnect a branch.
Detailed documentation is still a work in progress.

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

The fuel system is a connector option in FAST-OAD-CS23-HE, designed for fuel distribution with multiple inlets and outlets.
Detailed documentation is still a work in progress.

ID
**

.. code:: yaml

    fastga_he.pt_component.fuel_system


Hydrogen fuel system
====================

The hydrogen fuel system is a connector option in FAST-OAD-CS23-HE, designed for hydrogen distribution with multiple inlets and outlets.
Detailed documentation can be found at :ref:`h2-fuel-system`.

ID
**

.. code:: yaml

    fastga_he.pt_component.h2_fuel_system

Gearbox
=======

The gearbox is a connector option in FAST-OAD-CS23-HE, designed to transmit mechanical power between multiple components.
Detailed documentation is still a work in progress.

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

The planetary gear is a connector option in FAST-OAD-CS23-HE, designed to distribute mechanical power between multiple components.
Detailed documentation is still a work in progress.

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

The speed reducer is a connector option in FAST-OAD-CS23-HE, designed to reduce RPM with increasing torque output.
Detailed documentation is still a work in progress.

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

The DC load is a constant power load option in FAST-OAD-CS23-HE. 
Detailed documentation is still a work in progress.

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

The PMSM is a load option in FAST-OAD-CS23-HE, designed to convert electrical power in mechanical power. 
Detailed documentation is still a work in progress.

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

The propeller is a propulsor option in FAST-OAD-CS23-HE, designed to provide thrust for the aircraft.
Detailed documentation is still a work in progress.

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

The battery is a power source option in FAST-OAD-CS23-HE, designed to provide electricity.
Detailed documentation is still a work in progress.

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

The generator is a power source option in FAST-OAD-CS23-HE, designed to provide electricity from a mechanical power source.
Detailed documentation is still a work in progress.

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

The IC engine is a power source option in FAST-OAD-CS23-HE, designed to provide power by consuming AvGas or Diesel.
Detailed documentation is still a work in progress.

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

The high RPM IC engine is a power source option in FAST-OAD-CS23-HE, designed to provide power by consuming AvGas or Diesel.
Detailed documentation is still a work in progress.

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

The proton-exchange membrane fuel cell is a power source option in FAST-OAD-CS23-HE, designed to provide electricity by consuming hydrogen.
Detailed documentation can be found at :ref:`pemfc`.

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

The simple turbo generator is a power source option in FAST-OAD-CS23-HE, designed to provide electricity from turboshaft engine.
Detailed documentation is still a work in progress.

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

The turboshaft engine is a power source option in FAST-OAD-CS23-HE, designed to provide power by consuming Jet-A1.
Detailed documentation is still a work in progress.

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

The fuel tank is a storage tank option in FAST-OAD-CS23-HE, designed to carry AvGas or kerosene for the flight mission.
Detailed documentation is still a work in progress.

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

The gaseous hydrogen tank is a storage tank option in FAST-OAD-CS23-HE, designed to carry gaseous hydrogen for the flight mission.
Detailed documentation can be found at :ref:`gh2-tank`.

ID
**

.. code:: yaml

    fastga_he.pt_component.gaseous_hydrogen_tank

Constraints
***********

.. code:: yaml

    submodel.propulsion.constraints.gaseous_hydrogen_tank.capacity: fastga_he.submodel.propulsion.constraints.gaseous_hydrogen_tank.capacity.enforce

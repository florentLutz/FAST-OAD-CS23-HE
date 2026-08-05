.. _example-ampere:

=======================================================
AMPERE: 40-fan hybrid distributed electric propulsion
=======================================================

.. contents::

The reference aircraft
=======================

AMPERE is ONERA's distributed electric propulsion (DEP) concept aircraft: a high-wing CS-23
business aircraft with 40 electric ducted fans (EDF) mounted on the wing leading edge, also used
as a high-lift device through the fan slipstream (see the :ref:`ducted fan model <ducted-fan>`).
The hybrid powertrain combines PEMFC fuel cells and gaseous hydrogen tanks with batteries,
organized into 10 clusters of 4 fans each.

Main TLARs (:cite:`dillinger:2018`, Table 1): 4-6 passengers, 500 km range in about 2 hours,
STOL capability, FL100 ceiling (10,000 ft, unpressurized cabin).

This case (``ampere_final`` in the repository's ``integration_tests/``) reuses a Pipistrel's
structural geometry (fuselage, landing gear, tail arrangement) as a seed, replacing only the
propulsion system (40 EDF / 10 hybrid clusters) and the wing/tail/TLAR parameters with AMPERE's
published values. It is a validation of the propulsion architecture at AMPERE's scale, not a
complete, faithful sizing of the real aircraft -- see `Discussion and known limitations`_ below.

.. code:: yaml

    power_train_components:

      ducted_fan_1:
        id: fastga_he.pt_component.ducted_fan
        position: on_the_wing
      ⋮
      # 40 ducted fans total, organized into 10 clusters of 4,
      # each cluster fed by a PEMFC + H2 tank + battery pack

Main parameters: real vs. simulated
====================================

.. list-table::
   :header-rows: 1

   * - Parameter
     - Real (:cite:`dillinger:2018`, Table 1)
     - Simulated (``ampere_final``)
     - Difference
   * - MTOW
     - 2400 kg
     - 2180.3 kg
     - -9.2%
   * - Wing area
     - 25.925 m\ :sup:`2`
     - 27.97 m\ :sup:`2`
     - +7.9%
   * - Wing span
     - 14.5 m
     - 14.88 m
     - +2.6%
   * - HTP area
     - 3.8 m\ :sup:`2`
     - 2.99 m\ :sup:`2`
     - -21.4%
   * - VTP area
     - 2.02 m\ :sup:`2`
     - 1.95 m\ :sup:`2`
     - -3.6%
   * - Number of EDF
     - 40
     - 40
     - equal
   * - Installed power (motors + PEMFC)
     - 400 kW
     - ~296 kW
     - -26.0%
   * - Installed energy (battery + H2, LHV basis)
     - 500 kWh
     - ~432 kWh
     - -13.7%
   * - Range (main route)
     - 500 km
     - 500 km
     - equal (mission input)
   * - Payload
     - 4-6 PAX
     - 172 kg (not converted to PAX)
     - --

The wing and VTP converge much closer to the real values when the wing-sizing submodel is
switched to ``fastga_he.loop.wing_area`` (``UpdateWingAreaLiftDEPEquilibrium``), which solves a
landing/approach equilibrium accounting for the lift gain from each ducted fan's slipstream --
unlike the default ``fastga.loop.wing_area`` (a pure stall-speed/CL_max formula with no coupling
to propulsion), which inflated the wing by +41.5% in an earlier run with the same battery sizing.
The HTP swung to the undersized side as a side effect, likely because ``tail_sizing``/
``static_margin`` (still the default, non-DEP-aware loops) reacted to the CG shift caused by the
smaller wing (see `Discussion and known limitations`_).

Geometry
========

.. figure:: /img/examples/ampere/geometry_planform.png
   :width: 700
   :align: center

   Schematic sketch (not CAD-accurate) built from the span/chord/sweep values in the output XML.
   AMPERE's real wing (dashed rectangle) is overlaid for scale comparison. The 40 ducted fans are
   shown along the wing leading edge, matching the real configuration.

Mass breakdown
==============

.. figure:: /img/examples/ampere/mass_breakdown.png
   :width: 700
   :align: center

   The 2180 kg MTOW splits into airframe (542 kg), electric powertrain (1403 kg),
   furniture+systems (56 kg), and payload (172 kg). Within the powertrain, the battery is the
   dominant sub-system -- a direct reflection of the fixed battery-modules-per-cluster sizing (see
   `Discussion and known limitations`_) -- far heavier than the inverters, PEMFC stacks, or the
   ducted fans themselves.

Energy consumed over the mission
=================================

.. figure:: /img/examples/ampere/energy_profile.png
   :width: 700
   :align: center

   Total electric energy consumed over the mission: 240.5 kWh, split 133.3 kWh (55.4%) from the
   battery and 107.2 kWh (44.6%) from the PEMFC (electric, computed from H2 consumption x stack
   efficiency at each point). Mission phases (climb/cruise/descent/reserve) are shaded. The
   battery figure matches the value reported in the output XML (``energy_consumed_mission``
   summed over the 10 clusters), confirming the integration.

Battery: state of charge and C-rate
====================================

.. figure:: /img/examples/ampere/battery_soc_crate.png
   :width: 700
   :align: center

   SOC drops from 100% to 31.9% by the end of the mission (matching ``SOC_min`` reported in the
   XML). Maximum C-rate (0.46 1/h) occurs during climb. The min-max range across the 10 clusters
   essentially overlaps the mean, confirming the expected symmetry of the architecture.

Hydrogen consumed
==================

.. figure:: /img/examples/ampere/h2_mass_consumed.png
   :width: 700
   :align: center

   Total H2 mass consumed over the mission: 6.875 kg (10 tanks summed), out of a total onboard
   capacity of 7.08 kg. Consumption grows roughly linearly during cruise, with higher rates
   during climb.

Discussion and known limitations
=================================

* **Geometry origin:** fuselage, landing gear, and structural tail arrangement are still
  inherited from the Pipistrel; only the propulsion system and the main wing/tail/TLAR parameters
  were replaced with AMPERE's values. This is not a faithful, complete sizing of the real
  aircraft.
* **Installed power:** the motors+PEMFC add up to ~296 kW, versus the published 400 kW --
  likely because the simulated mission/aerodynamics (still influenced by the Pipistrel) don't
  demand as much peak thrust as the real aircraft would need for its STOL takeoff requirement,
  which this case doesn't reproduce as an active constraint.
* **Battery sizing:** ``SOC_min`` of 31.9% reflects the fixed battery-modules-per-cluster seed in
  ``run_ampere_case.py`` (``--battery-modules``, default 16), not an optimized sizing -- re-
  sweeping it lets you retarget a different SOC floor as the airframe geometry evolves.
* **HTP:** its size is a side effect of the smaller wing acting on the default (non-DEP-aware)
  ``tail_sizing``/``static_margin`` loops -- not investigated in depth in this case.
* **Literature note:** the "32 engines" figure that appears in :cite:`dillinger:2018` (Table 1,
  "Scale 1:5" column) refers to the 1:5-scale wind-tunnel mock-up (32 larger 50mm EDFs
  reproducing the thrust of 40 smaller 40mm EDFs at full scale), not an engine-out (OEI)
  redundancy requirement for the real aircraft.

Additional resources
=====================

An interactive HTML version of this report and the raw case outputs (``ampere_final_out.xml``,
``ampere_final_mission_data.csv``, ``ampere_final_power_train_data.csv``) are available in the
repository under ``integration_tests/ampere_final/``, generated by
``integration_tests/ampere_final/make_report.py`` from the case's converged run.

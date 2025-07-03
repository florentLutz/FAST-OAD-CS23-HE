.. _default-submodel:

====================
Registered submodels
====================

Default submodels registered for each FAST-OAD-CS23-HE execution, excluding the propulsion component constraint registry.
Propulsion components are documented separately under :ref:`propulsion components <propulsion-index>`.


.. raw:: html

   <div style="display: flex; justify-content: center;">
=================================================================  ===========================================================================================
Service                                                            Submodel
=================================================================  ===========================================================================================
submodel.aerodynamics.aircraft.max_level_speed             	       fastga.submodel.aerodynamics.aircraft.max_level_speed.legacy
submodel.aerodynamics.wing.slipstream.thrust_power_computation	   fastga.submodel.aerodynamics.wing.slipstream.thrust_power_computation.via_id
service.geometry.mfw	                                           fastga.submodel.geometry.mfw.legacy
service.geometry.fuselage.depth	                                   fastga.submodel.geometry.fuselage.depth.legacy
service.geometry.fuselage.volume	                               fastga.submodel.geometry.fuselage.volume.legacy
service.geometry.fuselage.wet_area	                               fastga.submodel.geometry.fuselage.wet_area.legacy
submodel.performances.mission.taxi	                               fastga.submodel.performances.mission.taxi.legacy
submodel.performances.mission.climb	                               fastga.submodel.performances.mission.climb.legacy
submodel.performances.mission.climb_speed	                       fastga.submodel.performances.mission.climb_speed.legacy
submodel.performances.mission.cruise	                           fastga.submodel.performances.mission.cruise.legacy
submodel.performances.mission.descent	                           fastga.submodel.performances.mission.descent.legacy
submodel.performances.mission.descent_speed	                       fastga.submodel.performances.mission.descent_speed.legacy
submodel.performances.mission.reserves	                           fastga.submodel.performances.mission.reserves.legacy
submodel.loop.wing_area.update.aero	                               fastga.submodel.loop.wing_area.update.aero.simple
submodel.loop.wing_area.constraint.aero	                           fastga.submodel.loop.wing_area.constraint.aero.simple
submodel.loop.wing_area.update.geom	                               fastga.submodel.loop.wing_area.update.geom.simple
submodel.loop.wing_area.constraint.geom	                           fastga.submodel.loop.wing_area.constraint.geom.simple
submodel.performances.dep_effect	                               fastga.submodel.performances.dep_effect.none
submodel.performances.energy_consumption	                       fastga.submodel.performances.energy_consumption.ICE
submodel.weight.cg.airframe.landing_gear	                       fastga.submodel.weight.cg.airframe.landing_gear.legacy
service.weight.mass.airframe.wing	                               fastga.submodel.weight.mass.airframe.wing.legacy
service.weight.mass.airframe.fuselage	                           fastga.submodel.weight.mass.airframe.fuselage.legacy
service.weight.mass.airframe.flight_controls	                   fastga.submodel.weight.mass.airframe.flight_controls.legacy
service.weight.mass.airframe.landing_gear	                       fastga.submodel.weight.mass.airframe.landing_gear.legacy
service.weight.mass.airframe.paint	                               fastga.submodel.weight.mass.airframe.paint.no_paint
service.weight.mass.airframe.tail	                               fastga.submodel.weight.mass.airframe.tail.legacy
service.weight.mass.propulsion.installed_engine	                   fastga.submodel.weight.mass.propulsion.installed_engine.legacy
service.weight.mass.propulsion.fuel_system	                       fastga.submodel.weight.mass.propulsion.fuel_system.legacy
service.weight.mass.propulsion.unusable_fuel	                   fastga.submodel.weight.mass.propulsion.unusable_fuel.legacy
service.weight.mass.system.life_support_system	                   fastga.submodel.weight.mass.system.life_support_system.legacy
service.weight.mass.system.avionics_system	                       fastga.submodel.weight.mass.system.avionics_systems.legacy
service.weight.mass.system.recording_system	                       fastga.submodel.weight.mass.system.recording_systems.minimum
submodel.aerodynamics.nacelle.cd0	                               fastga_he.submodel.aerodynamics.powertrain.cd0.from_pt_file
submodel.propulsion.performances.dc_line.temperature_profile	   fastga_he.submodel.propulsion.performances.dc_line.temperature_profile.constant
submodel.propulsion.inverter.junction_temperature	               fastga_he.submodel.propulsion.inverter.junction_temperature.fixed
submodel.propulsion.rectifier.junction_temperature	               fastga_he.submodel.propulsion.rectifier.junction_temperature.fixed
submodel.performances_he.dep_effect	                               fastga_he.submodel.performances.dep_effect.from_pt_file
submodel.performances_he.energy_consumption	                       fastga_he.submodel.performances.energy_consumption.from_pt_file
submodel.propulsion.cg	                                           fastga_he.submodel.propulsion.cg.from_pt_file
submodel.propulsion.delta_cd	                                   fastga_he.submodel.propulsion.delta_cd.from_pt_file
submodel.propulsion.delta_cl	                                   fastga_he.submodel.propulsion.delta_cl.from_pt_file
submodel.propulsion.delta_cm	                                   fastga_he.submodel.propulsion.delta_cm.from_pt_file
submodel.propulsion.drag	                                       fastga_he.submodel.propulsion.drag.from_pt_file
submodel.propulsion.mass	                                       fastga_he.submodel.propulsion.mass.from_pt_file
submodel.propulsion.performances	                               fastga_he.submodel.propulsion.performances.from_pt_file
submodel.propulsion.thrust_distributor	                           fastga_he.submodel.propulsion.thrust_distributor.legacy
submodel.propulsion.wing.punctual_loads	                           fastga_he.submodel.propulsion.wing.punctual_loads.from_pt_file
submodel.propulsion.wing.punctual_tanks	                           fastga_he.submodel.propulsion.wing.punctual_tanks.from_pt_file
submodel.propulsion.wing.distributed_loads	                       fastga_he.submodel.propulsion.wing.distributed_loads.from_pt_file
submodel.propulsion.wing.distributed_tanks	                       fastga_he.submodel.propulsion.wing.distributed_tanks.from_pt_file
submodel.propulsion.performances.dc_line.resistance_profile	       fastga_he.submodel.propulsion.performances.dc_line.resistance_profile.from_temperature
submodel.propulsion.sizing.dc_line.length	                       fastga_he.submodel.propulsion.sizing.dc_line.length.from_position
submodel.propulsion.dc_dc_converter.efficiency	                   fastga_he.submodel.propulsion.dc_dc_converter.efficiency.fixed
submodel.propulsion.dc_dc_converter.weight	                       fastga_he.submodel.propulsion.dc_dc_converter.weight.sum
submodel.propulsion.inverter.efficiency	                           fastga_he.submodel.propulsion.inverter.efficiency.fixed
submodel.propulsion.rectifier.efficiency	                       fastga_he.submodel.propulsion.rectifier.efficiency.fixed
submodel.propulsion.rectifier.weight	                           fastga_he.submodel.propulsion.rectifier.weight.sum
service.propulsion.battery.lifespan	                               None
submodel.weight.cg.propulsion	                                   fastga_he.submodel.weight.cg.propulsion.power_train
submodel.weight.cg.loadcase.flight	                               fastga.submodel.weight.cg.loadcase.flight.legacy
submodel.performances.cg_variation	                               fastga_he.submodel.performances.cg_variation.legacy
service.weight.mass.propulsion	                                   fastga_he.submodel.weight.mass.propulsion.power_train
submodel.weight.mass_breakdown	                                   fastga.submodel.weight.mass_breakdown.legacy
service.weight.mass.payload	                                       fastga.submodel.weight.mass.payload.legacy
submodel.weight.mass.mzfw_and_mlw	                               fastga_he.submodel.weight.mass.mzfw_and_mlw.legacy
=================================================================  ===========================================================================================

.. raw:: html

   </div>

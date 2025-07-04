.. _default-submodel:

====================
Registered submodels
====================

To ensure compatibility between the aircraft design model from FAST-GA and the powertrain file mechanic introduced by
FAST-OAD-CS23-HE, a certain number of submodel are enabled by default and listed here. Submodels related to propulsion
components are documented separately under :ref:`propulsion components <propulsion-index>`.


.. raw:: html

   <div style="display: flex; justify-content: center;">

=================================================================  ===========================================================================================
Service                                                            Submodel
=================================================================  ===========================================================================================
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
submodel.weight.cg.propulsion	                                   fastga_he.submodel.weight.cg.propulsion.power_train
submodel.performances.cg_variation	                               fastga_he.submodel.performances.cg_variation.legacy
service.weight.mass.propulsion	                                   fastga_he.submodel.weight.mass.propulsion.power_train
=================================================================  ===========================================================================================

.. raw:: html

   </div>

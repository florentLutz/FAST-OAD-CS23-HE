.. _rta_integration:

===============
RTA integration
===============
`FAST-OAD-RTA <https://github.com/fast-aircraft-design/RTA>`_ is a FAST-OAD plugin for designing and analyzing Regional
Transport Aircraft. To ensure smooth communication between FAST-OAD-RTA and FAST-OAD-CS23-HE, the models
and submodels specified in the following ATR aircraft `configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_
template needs to be added after installing FAST-OAD-RTA.


.. code:: yaml

    model:
    ⋮
      subgroup:
        ⋮
        rta_integration:
          id: fastga_he.rta_variables

        performances:
          id: fastga_he.performances.mission_vector
          ⋮
          rta_activation: True

    submodels:
      service.aerodynamics.CD0.sum: fastoad.submodel.aerodynamics.CD0.sum.rta
      service.aerodynamics.CD0.wing: fastoad.submodel.aerodynamics.CD0.wing.rta
      service.aerodynamics.polar: fastoad.submodel.aerodynamics.polar.rta
      service.cg.empty_aircraft: fastoad.submodel.weight.cg.empty_aircraft.rta
      service.mass.propulsion: fastga_he.submodel.weight.mass.propulsion.engine.power_train.rta
      # Apply only for retrofitting
      service.performances.delta_m: fastga_he.submodel.performances.delta_m.set_value.retrofit.rta
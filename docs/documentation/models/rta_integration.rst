.. _rta_integration:

===============
RTA integration
===============
`FAST-OAD-RTA <https://github.com/fast-aircraft-design/RTA>`_ is a FAST-OAD plugin for designing and analyzing Regional
Transport Aircraft. To ensure smooth communication between the OAD models from RTA and the powertrain models from FAST-OAD-CS23-HE, the models and submodels specified in the
ATR aircraft `configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_
template are required to be added.


.. code:: yaml

    model:
    ⋮
      subgroup:
        ⋮
        rta_integration:
          id: fastga_he.rta_variables
          power_train_file_path: pt_file.yml

        performances:
          id: fastga_he.performances.mission_vector


    submodels:
      service.aerodynamics.CD0.sum: fastga_he.submodel.aerodynamics.sum.cd0.rta
      service.aerodynamics.CD0.wing: fastga_he.submodel.aerodynamics.wing.cd0.rta
      service.aerodynamics.polar: fastoad.submodel.aerodynamics.polar.legacy
      service.cg.propulsion: fastga_he.submodel.weight.cg.nacelle.rta
      service.mass.propulsion: fastga_he.submodel.weight.mass.propulsion.power_train.rta
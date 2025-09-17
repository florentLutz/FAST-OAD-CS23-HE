.. _rta_integration:

===============
RTA integration
===============
`FAST-OAD-RTA <https://github.com/fast-aircraft-design/RTA>`_ is a FAST-OAD plugin for designing and analyzing Regional
Transport Aircraft. To enable smooth communication between FAST-OAD-RTA and FAST-OAD-CS23-HE input variables, include
the ``fastga_he.rta_variables`` model and set the ``rta_activation`` option under
``fastga_he.performances.mission_vector`` (for full-sizing missions) or
``fastga_he.performances.operational_mission_vector`` (for operational missions) in the ATR aircraft `configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_
after installing FAST-OAD-RTA.


.. code:: yaml

    rta_integration:
      id: fastga_he.rta_variables

    performances:
      id: fastga_he.performances.mission_vector
      ⋮
      rta_activation: True
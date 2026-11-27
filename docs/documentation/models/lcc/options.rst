.. _options-lca:

=======================
Life Cycle Cost options
=======================

The Life Cycle Cost (LCC) module can be parametrized according to several criterion which are implemented under
the form of group options. A description of those options is available here.

.. code:: yaml

    model:
        lcc:
            id: fastga_he.lcc.legacy
            power_train_file_path: hybrid_propulsion.yml
            delivery_method: Train # Train, Flight
            loan: True # True, False
            years_of_development: 10
            years_of_program: 30
            years_of_service: 30
            profitability_analysis: False # True, False

| ``power_train_file_path``: The path to the powertrain architecture configuration file of the run
| ``delivery_method`` : Detail description can be found at :ref:`LCA delivery method <aircraft_delivery>`.
| ``loan`` : Activate if loan is one of the financial source during purchase, set to ``True`` by default.
| ``years_of_development`` : The number of years of development of the aircraft, default is set to 10 years.
| ``years_of_program`` : The number of years of program of the aircraft, default is set to 30 years.
| ``years_of_service`` : The number of years of service of the aircraft, default is set to 30 years.
| ``profitability_analysis`` : Activate if a profitability analysis is required, set to ``False`` by default.

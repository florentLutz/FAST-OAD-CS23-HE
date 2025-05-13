.. _options-lca:

=======================
Life Cycle Cost options
=======================

The Life Cycle Cost (LCC) module can be parametrized according to several criterion which are implemented under
the form of group options. A description of those options are available here.

.. code:: yaml

    model:
        lcc:
            id: fastga_he.lcc.legacy
            power_train_file_path: hybrid_propulsion.yml
            delivery_method: Train # Train, Flight
            loan: True # True, False


| ``power_train_file_path``: The path to the powertrain architecture configuration file of the run
| ``delivery_method`` : Detail description can be found at :ref:`LCA delivery method <aircraft_delivery>`.
| ``loan`` : Activate if loan is one of the financial source during purchase, set to ``True`` by default.

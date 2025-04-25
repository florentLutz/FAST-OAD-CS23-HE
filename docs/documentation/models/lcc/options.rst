.. _options-lca:

=============================
Life Cycle Cost options
=============================

The Life Cycle Cost (LCC) module can be parametrized according to several criterion which are implemented under
the form of group options. A description of those options is available here.



****************************
Powertrain file path options
****************************

The LCC module computes the purchase cost, annual operation cost, and the annual fuel / electricity cost of the
aircraft powertrain based on the components specified in the powertrain architecture. The path to that file is given in
the :code:`power_train_file_path` option.

.. code:: yaml

    model:
        lcc:
            id: fastga_he.lcc.legacy
            power_train_file_path: hybrid_propulsion.yml



************
Cost options
************
The LCC of an aircraft breakdown into two main groups, the production costs (including purchase cost and freight)
and the yearly operation expense of the aircraft. By default these options (except the :code:`loan` option) are set to
:code:`False`.

.. code:: yaml

    model:
        lcc:
            id: fastga_he.lcc.legacy
            power_train_file_path: hybrid_propulsion.yml
            option_name: True


Production cost options
=======================

These are the production cost options based on different aircraft configurations:

| ``complex_flap`` : Activate if the complex flap system is configured for the wing.
| ``pressurized`` : Activate if the cabin is pressurized.
| ``tapered_wing`` : Activate if the taper-wing design is applied.


Operation cost options
======================

These are the operation cost options based on purchase financing and variable naming consistency:

| ``loan`` : Activate if loan is one of the financial source during purchase.
| ``use_operational_mission`` : LCA model inheritance ensures consistent variable naming.


.. _powertrain-builder-index:

==================
Powertrain builder
==================

As an unique feature of FAST-OAD-CS23-HE, the powertrain builder allow high flexibility for user to explore wide range of
aircraft powertrain architectures. To active this module, simply set the ``power_train_configuration_file path`` option to
the model(s) with either or both of these two IDs, ``fastga_he.power_train.sizing`` or ``fastga_he.performances.mission_vector``
, in the `configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_.

.. code:: yaml

  power_train_sizing:
    id: fastga_he.power_train.sizing
    power_train_file_path: ./<powertrain_config>.yml
  performances:
    id: fastga_he.performances.mission_vector
    ⋮
    power_train_file_path: ./<powertrain_config>.yml
    sort_component: true

A description of the powertrain customization is available here. It includes a description of powertrain configuation
file, lists of the components available with their constraints, a description of powertrain network visualization, and
simple powertrain templates.

.. toctree::
   :maxdepth: 2

    Powertrain configuration (PT) file <pt_file>
    Component Constraints and ID <constraints>
    PT file visualization <pt_visual>
    PT file templates <templates>

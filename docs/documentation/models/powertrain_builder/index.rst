===================
Powertrain building
===================

FAST-OAD-CS23-HE allows to configure powertrain architectures of hybrid-electric aircraft via a powertrain builder
module. This module can be integrated to the analysis by defining the ``power_train_configuration_file path`` option to
power_train_sizing or performances model if registered with the corresponding `id` in
`configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_.

.. code:: yaml

  power_train_sizing:
    id: fastga_he.power_train.sizing
    power_train_file_path: ./<powertrain_config>.yml
  performances:
    id: fastga_he.performances.mission_vector
    ⋮
    power_train_file_path: ./<powertrain_config>.yml

A description of the LCC module is available here. It includes a description of the models, a description of the options
available as well as the value they can take and a description of some of the key assumptions that were made.

.. toctree::
   :maxdepth: 2

    Powertrain configuration (PT) file <pt_file>
    PT file templates <templates>
    Component Constraints <constraints>
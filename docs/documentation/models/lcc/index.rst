======================
Life Cycle Cost models
======================

FAST-OAD-CS23-HE allows to perform a Life Cycle Cost analysis of hybrid-electric aircraft via a LCC module. This module
can be integrated to the analysis by adding the corresponding `id` to the `configuration file <https://fast-oad.readthedocs.io/en/stable/documentation/usage.html#problem-definition>`_.
The `id` for the LCC module is:

.. code:: yaml

    id: fastga_he.lcc.legacy


A description of the LCC module is available here. It includes a description of the models, a description of the options
available as well as the value they can take and a description of some of the key assumptions that were made.

.. toctree::
   :maxdepth: 2

    LCC module functioning <models>
    Options description <options>
    LCC module assumptions <assumptions>
============================
Life Cycle Assessment models
============================

FAST-OAD-CS23-HE allows to perform a Life Cycle Assessment of hybrid-electric aircraft via a LCA module. That module can be integrated to the analysis by adding the corresponding `id` to the `configuration file <https://fast-oad.readthedocs.io/en/v1.8.2/documentation/usage.html#problem-definition>`_. The id for the LCA module is:

.. code:: yaml

    id: fastga_he.lca.legacy


A description of this LCA module is available here. It includes a description of the models, a description of the option available as well as the value they can take and a description of some of the key assumptions that were made.

.. toctree::
   :maxdepth: 2

    LCA module functioning <models>
    Options description <options>
    LCA module assumptions <assumptions>
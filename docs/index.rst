===========================
Welcome to FAST-OAD-CS23-HE
===========================

For a description of the features of the FAST-OAD framework, you may explore the `official documentation <https://fast-oad.readthedocs.io/en/stable/>`_

For a description of models used in FAST-OAD-CS23-HE, you may look at
:ref:`models-index`.

.. note::

    Models in FAST-OAD-CS23-HE are still a work in progress.


Powertrain builder
==================

The powertrain builder is a novel feature exclusive to FAST-OAD-CS23-HE, enabling flexible configuration of aircraft
powertrain architectures. It offers high flexibility in the definition of the propulsive system. To active this module, simply set the ``power_train_configuration_file path`` option to
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

For further details of powertrain builder, you may look at :ref:`powertrain-builder-index`.


Contents
========

.. toctree::
   :maxdepth: 1

   License <license>
   Authors <authors>
   Powertrain Builder <pt_builder>
   Citation <citation>
   Changelog <changelog>
   API Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists

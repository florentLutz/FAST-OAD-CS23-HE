.. _attributes:

=========================================
PT file visualization function attributes
=========================================
Other than the mandatory settings like the PT config file path and where you want to save the HTML, you can tweak several other
options to adjust how the final plot looks.

Sorting
=======
This attributes is used to activate the sorting mechanism, set to `True` as default to enable sorting. To disable,
use:

.. code:: python

    power_train_network_viewer(pt_file_path, network_file_path, sorting=False)

Sorting reference layer
=======================
This attribute sets the :ref:`reference level <ref-level-sort>` used by the sorting mechanism. The default approach is
the `from-storage approach`, which helps minimise connection crossings and possibly improves the structure layout.

.. caution::

    If there is any non-propulsor node at the top of the hierarchy, the attribute will be overwritten with
    `from-propulsor approach` to prevent crossing.

To configure the sorting process to `from-propulsor approach`, use:

.. code:: python

    power_train_network_viewer(pt_file_path, network_file_path, from_propulsor=True)

.. note::

    If the crossing remains after switching between two approaches, please check the the sequencing of the node(s)
    that is/are causing crossing in the PT configuration file.

Plot orientation
================
This attribute specifies the orientation of the main powertrain architecture. The four valid options are
``"TB"`` (Top–Bottom), ``"BT"`` (Bottom–Top), ``"LR"`` (Left–Right), and ``"RL"`` (Right–Left). To configure the
orientation, use:

.. code:: python

    power_train_network_viewer(pt_file_path, network_file_path, orientation="TB")

Legend position
===============
This attribute specifies the placement of the connection-color legend. Valid entries are:
T (top), M (middle, vertical), B (bottom), L (left), R (right), and C (center, horizontal).

The layout is illustrated below:

.. image:: ../../../img/legend_position.svg
    :width: 400
    :align: center


To configure the position, use:

.. code:: python

    power_train_network_viewer(pt_file_path, network_file_path, legend_position="TR")

Plot size scaling
=================
This attribute allows to resize the architecture with keeping the original aspect ratio. Only positive
values are valid. To configure this factor, use:

.. code:: python

    power_train_network_viewer(pt_file_path, network_file_path, plot_scaling=1.2)

Legend size scaling
===================
This attribute allows to resize the color legend with keeping the original aspect ratio. Only positive
values are valid. To configure this factor, use:

.. code:: python

    power_train_network_viewer(pt_file_path, network_file_path, legend_scaling=1.2)

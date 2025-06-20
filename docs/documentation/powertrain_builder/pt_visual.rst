.. _visual:

=====================
PT file visualization
=====================
The ``power_train_network_viewer`` function generates a visual representation of a powertrain architecture based on a PT
file. The example below demonstrates how to import and use this function with the single ICE architecture from the
:ref:`PT file templates <templates>`.


.. code:: python

    from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

    pt_file_path = "/path/to/powertrain/configurations/PT_config.yml"
    network_file_path = "/path/to/network/topology/component_connections.html"

    power_train_network_viewer(pt_file_path, network_file_path)

    # pt_file_path: path to the PT configuration file
    # network_file_path: path to save the generated powertrain network visualization

The generated network should look like this:

.. raw:: html

   <a href="../../../../visual/single_ice.html" target="_blank">Single ICE architecture visualization</a><br>


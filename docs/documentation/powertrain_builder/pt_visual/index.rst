.. _visual:

=====================
PT file visualization
=====================
The ``power_train_network_viewer`` function generates a visual representation of a powertrain architecture based on the PT
file. The example below demonstrates how to import and use this function with the single ICE architecture from the
:ref:`PT file templates <templates>`.

.. toctree::
   :maxdepth: 1

    Viewer settings <attributes>
    Sorting Mechanism <sort_mechanism>


.. code:: python

    from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

    pt_file_path = "/path/to/powertrain/configurations/PT_config.yml"
    network_file_path = "/path/to/network/topology/component_connections.html"
    # path to save the generated powertrain network visualization

    power_train_network_viewer(pt_file_path, network_file_path)

The generated network should look like this:

.. raw:: html

   <a href="../../../single_ice.html" target="_blank" style="text-decoration: none;">
       <img src="../../../_images/single_ice.svg" alt="Single ICE powertrain architecture" style="max-width: 100%; cursor: pointer; opacity: 1; transition: opacity 0.3s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
   </a>
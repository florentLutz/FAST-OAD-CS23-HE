# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os

from IPython.core.display import display, HTML

import networkx as nx
from pyvis.network import Network

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator


def power_train_network_viewer(
    power_train_file_path: str,
    network_file_path: str,
):

    # Notebook is at True to prevent him from opening a browser
    net = Network(notebook=True)
    # Create directed graph object
    graph = nx.DiGraph()

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)

    names, connections = configurator.get_network_elements_list()

    for component_name in names:
        graph.add_node(component_name)

    for connection in connections:
        # When the component is connected to a bus, the output number is also specified but it
        # isn't meaningful when drawing a graph, so we will just filter it
        if type(connection[0]) is list:
            source = connection[0][0]
        else:
            source = connection[0]

        if type(connection[1]) is list:
            target = connection[1][0]
        else:
            target = connection[1]

        graph.add_edge(source, target)

    # The drawback of this method is that the creation of directories and file is not controled,
    # it is created based on the working directory. So we will have to do some shenanigans to get
    # and change the working directory once we are done with the creation of the graph.

    directory_to_save_graph = os.path.dirname(network_file_path)
    graph_name = os.path.basename(network_file_path)
    old_working_directory = os.getcwd()

    # Set the new working directory
    os.chdir(directory_to_save_graph)

    net.from_nx(graph)
    net.show(graph_name)
    display(HTML(graph_name))

    # Change the working directory back
    os.chdir(old_working_directory)

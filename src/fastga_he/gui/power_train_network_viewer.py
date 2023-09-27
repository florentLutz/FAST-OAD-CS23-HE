# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import networkx as nx
from pyvis.network import Network

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from . import icons

# Image URLs for graph nodes

icons_dict = {
    "battery": pth.join(icons.__path__[0], "battery.png"),
    "bus_bar": pth.join(icons.__path__[0], "bus_bar.png"),
    "cable": pth.join(icons.__path__[0], "cable.png"),
    "e_motor": pth.join(icons.__path__[0], "e_motor.png"),
    "generator": pth.join(icons.__path__[0], "generator.png"),
    "ice": pth.join(icons.__path__[0], "ice.png"),
    "switch": pth.join(icons.__path__[0], "switch.png"),
    "propeller": pth.join(icons.__path__[0], "propeller.png"),
    "splitter": pth.join(icons.__path__[0], "splitter.png"),
    "rectifier": pth.join(icons.__path__[0], "AC_DC.png"),
    "dc_converter": pth.join(icons.__path__[0], "DC_DC.png"),
    "inverter": pth.join(icons.__path__[0], "DC_AC.png"),
    "fuel_tank": pth.join(icons.__path__[0], "fuel_tank.png"),
    "fuel_system": pth.join(icons.__path__[0], "fuel_system.png"),
}


def power_train_network_viewer(
    power_train_file_path: str,
    network_file_path: str,
):

    # Notebook is at True to prevent him from opening a browser
    net = Network(
        notebook=True,
        cdn_resources="remote",
        bgcolor="#bebebe",
        font_color="white",
        select_menu=True,
        layout=True,
    )
    # Create directed graph object
    graph = nx.DiGraph()

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)

    (
        names,
        connections,
        components_type,
        icons_name,
        icons_size,
    ) = configurator.get_network_elements_list()
    distance_from_prop_loads, prop_loads = configurator.get_distance_from_propulsive_load()

    for component_name, component_type, icon_name, icon_size in zip(
        names, components_type, icons_name, icons_size
    ):
        if "propulsor" in component_type:
            level = 0
        elif ("propulsive_load" in component_type) and (component_name in prop_loads):
            level = 1
        elif "source" in component_type:
            if component_name in prop_loads:
                level = 1
            else:
                level = distance_from_prop_loads[component_name] + 1
        else:
            level = distance_from_prop_loads[component_name] + 1

        graph.add_node(
            component_name,
            value=icon_size,
            level=level,
            shape="image",
            image="file://" + icons_dict[icon_name],
        )

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

    # Set the new working directory and create it if it doesn't exist
    if not os.path.exists(directory_to_save_graph):
        os.makedirs(directory_to_save_graph)

    os.chdir(directory_to_save_graph)

    net.from_nx(graph)
    net.show(graph_name)

    # Change the working directory back
    os.chdir(old_working_directory)

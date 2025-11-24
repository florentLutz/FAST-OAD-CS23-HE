# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import scipy.sparse as sci_sparse


class HierarchicalLayout:
    """
    Implements the hierarchical graph layout algorithm.

    The algorithm has three main phases:
    1. Layer assignment (places nodes into layers using flow_hierarchy)
    2. Crossing minimization (reduces edge crossings between layers using Tutte's algorithm)
    3. Coordinate assignment (positions nodes within layers)
    """

    def __init__(self, graph, orientation="TB", node_layer_dict=None):
        """
        Initialize the hierarchical layout.

        Args:
            graph: NetworkX DiGraph
            orientation: 'TB' (top-bottom), 'BT' (bottom-top), 'LR' (left-right), 'RL' (right-left)
            node_layer_dict: Dictionary mapping nodes to their layer assignment (required)
        """
        self.graph = graph
        self.orientation = orientation
        self.layers = []
        self.node_layer = {}
        self.positions = {}
        self.node_layer_dict = node_layer_dict

    def _assign_layers(self):
        """Assign each node to a layer using provided override with flow_hierarchy as fallback."""
        self.node_layer = {node: int(layer) for node, layer in self.node_layer_dict.items()}

        max_layer = max(self.node_layer.values()) if self.node_layer else 0
        self.layers = [[] for _ in range(max_layer + 1)]

        for node, layer in self.node_layer.items():
            self.layers[layer].append(node)

        return self.layers

    def _minimize_crossings(self):
        """Minimize edge crossings using Tutte's algorithm."""
        self.positions = _resolve_crossings_with_tutte(self.graph, self.layers, self.orientation)

    def _assign_coordinates(self, layer_spacing=100, node_spacing=100):
        """Assign x, y coordinates to nodes based on orientation."""
        # If Tutte's algorithm was applied, positions already exist
        if self.positions:
            return self.positions

        # Otherwise use traditional coordinate assignment
        self.positions = {}
        max_layer = len(self.layers) - 1

        for layer_idx, nodes in enumerate(self.layers):
            num_nodes = len(nodes)

            if self.orientation == "TB":
                y = max_layer * layer_spacing - layer_idx * layer_spacing
                x_offset = -(num_nodes - 1) * node_spacing / 2

                for i, node in enumerate(nodes):
                    x = x_offset + i * node_spacing
                    self.positions[node] = (x, y)

            elif self.orientation == "BT":
                y = layer_idx * layer_spacing
                x_offset = -(num_nodes - 1) * node_spacing / 2

                for i, node in enumerate(nodes):
                    x = x_offset + i * node_spacing
                    self.positions[node] = (x, y)

            elif self.orientation == "LR":
                x = layer_idx * layer_spacing
                y_offset = -(num_nodes - 1) * node_spacing / 2

                for i, node in enumerate(nodes):
                    y = y_offset + i * node_spacing
                    self.positions[node] = (x, y)

            elif self.orientation == "RL":
                x = max_layer * layer_spacing - layer_idx * layer_spacing
                y_offset = -(num_nodes - 1) * node_spacing / 2

                for i, node in enumerate(nodes):
                    y = y_offset + i * node_spacing
                    self.positions[node] = (x, y)

        return self.positions

    def compute(self, layer_spacing=100, node_spacing=100):
        """Compute the layout by running all phases of the algorithm."""
        self._assign_layers()
        self._minimize_crossings()
        # If Tutte's algorithm didn't already assign coordinates, do it now
        if not self.positions:
            self._assign_coordinates(layer_spacing, node_spacing)

        return self.positions


def _resolve_crossings_with_tutte(
    graph, layers, orientation="TB", layer_spacing=100, node_spacing=100
):
    """
    Apply Tutte's drawing algorithm :cite:`tutte:1963` to minimize edge crossings
    in hierarchical layout.

    The top layer and bottom layer are fixed as boundary nodes, while interior layers
    are positioned to minimize edge crossings.

    Args:
        graph: NetworkX DiGraph
        layers: List of nodes per layer
        orientation: Graph orientation ('TB', 'BT', 'LR', 'RL')
        layer_spacing: Spacing between layers
        node_spacing: Spacing between nodes in a layer

    Returns:
        Dictionary mapping nodes to (x, y) coordinates
    """
    # Return the node set object of Networkx into ordinary list
    all_nodes = list(graph.nodes())
    num_layers = len(layers)

    # Identify boundary nodes (first and last layers)
    boundary_nodes = layers[0] + layers[-1]
    interior_nodes = [n for n in all_nodes if n not in boundary_nodes]

    # Create node to index mapping
    interior_idx = {node: i for i, node in enumerate(interior_nodes)}
    num_interior_nodes = len(interior_nodes)

    # Build adjacency list for each interior node
    adjacency_dict = {node: [] for node in all_nodes}  # generate empty lists for each interior node
    # Adding the neighbor connected nodes into the list
    for u, v in graph.edges():  #
        adjacency_dict[u].append(v)
        adjacency_dict[v].append(u)

    # Assign boundary coordinates based on layer positions
    boundary_coords = {}

    # First layer (top/left boundary) setup
    top_nodes = layers[0]
    num_top = len(top_nodes)
    for i, node in enumerate(top_nodes):
        if orientation in ["TB", "BT"]:
            x = -(num_top - 1) * node_spacing / 2 + i * node_spacing
            y = -1.0
        else:  # LR, RL
            x = -1.0
            y = -(num_top - 1) * node_spacing / 2 + i * node_spacing
        boundary_coords[node] = (x, y)

    # Last layer (bottom/right boundary) setup
    bottom_nodes = layers[-1]
    num_bottom = len(bottom_nodes)
    for i, node in enumerate(bottom_nodes):
        if orientation in ["TB", "BT"]:
            x = -(num_bottom - 1) * node_spacing / 2 + i * node_spacing
            y = 1.0
        else:  # LR, RL
            x = 1.0
            y = -(num_bottom - 1) * node_spacing / 2 + i * node_spacing
        boundary_coords[node] = (x, y)

    # Build Laplacian matrix for interior nodes
    # Laplacian matrix is the A matrix of the AX = b_x , AY = b_y linear system of equations
    laplacian_matrix = np.zeros((num_interior_nodes, num_interior_nodes))
    b_x = np.zeros(num_interior_nodes)
    b_y = np.zeros(num_interior_nodes)

    for node in interior_nodes:
        i = interior_idx[node]
        neighbors = adjacency_dict[node]
        degree = len(neighbors)

        if degree == 0:
            continue
        # Fill up the diagonal with the amount of edges
        laplacian_matrix[i, i] = degree

        for neighbor in neighbors:
            # For the connected neighbor node
            if neighbor in interior_idx:
                j = interior_idx[neighbor]
                laplacian_matrix[i, j] -= 1

            else:
                # Boundary node contribution (directly assign value as the coordinate is fixed)
                bx, by = boundary_coords[neighbor]
                b_x[i] += bx
                b_y[i] += by

    # Solve linear systems
    try:
        laplacian_sparse = sci_sparse.csr_matrix(laplacian_matrix)
        x_interior = sci_sparse.linalg.spsolve(laplacian_sparse, b_x)
        y_interior = sci_sparse.linalg.spsolve(laplacian_sparse, b_y)
    except Exception:
        # If solver fails, return empty dict to fall back to traditional layout
        return {}

    # Construct final coordinates
    coords = boundary_coords.copy()
    for node, i in interior_idx.items():
        coords[node] = (float(x_interior[i]), float(y_interior[i]))

    # Convert Tutte coordinates to hierarchical layer coordinates
    # Map nodes back to layer positions based on their Tutte y-coordinate (or x for LR/RL)
    positions = {}

    for layer_idx, layer_nodes in enumerate(layers):
        if orientation in ["TB", "BT"]:
            # y coordinate determines vertical position
            layer_y = (
                layer_spacing * layer_idx
                if orientation == "BT"
                else layer_spacing * (num_layers - 1 - layer_idx)
            )

            # Sort nodes by their x-coordinate from Tutte
            sorted_nodes = sorted(layer_nodes, key=lambda n: coords[n][0])
            num_nodes = len(sorted_nodes)
            x_offset = -(num_nodes - 1) * node_spacing / 2

            for i, node in enumerate(sorted_nodes):
                x = x_offset + i * node_spacing
                positions[node] = (x, layer_y)
        else:  # LR, RL
            # x coordinate determines horizontal position
            layer_x = (
                layer_spacing * layer_idx
                if orientation == "LR"
                else layer_spacing * (num_layers - 1 - layer_idx)
            )

            # Sort nodes by their y-coordinate from Tutte
            sorted_nodes = sorted(layer_nodes, key=lambda n: coords[n][1])
            num_nodes = len(sorted_nodes)
            y_offset = -(num_nodes - 1) * node_spacing / 2

            for i, node in enumerate(sorted_nodes):
                y = y_offset + i * node_spacing
                positions[node] = (layer_x, y)

    return positions

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import networkx as nx


class HierarchicalLayout:
    """
    Implements the hierarchical graph layout algorithm.

    The algorithm has three main phases:
    1. Layer assignment (places nodes into layers using flow_hierarchy)
    2. Crossing minimization (reduces edge crossings between layers)
    3. Coordinate assignment (positions nodes within layers)
    """

    def __init__(self, graph, orientation="TB", node_layer_dict=None):
        """
        Initialize the hierarchical layout.

        Args:
            graph: NetworkX DiGraph
            orientation: 'TB' (top-bottom), 'BT' (bottom-top),
                        'LR' (left-right), 'RL' (right-left)
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
        if self.node_layer_dict:
            # assign layers based on the distance from energy storage component nodes
            self.node_layer = {node: int(layer) for node, layer in self.node_layer_dict.items()}
        else:
            # Fallback in case node layer is not provided
            self.node_layer = nx.algorithms.dag_longest_path_length(self.graph)

        max_layer = max(self.node_layer.values()) if self.node_layer else 0
        self.layers = [[] for _ in range(max_layer + 1)]

        for node, layer in self.node_layer.items():
            self.layers[layer].append(node)

        return self.layers

    def _minimize_crossings(self):
        """Reorder nodes within layers to minimize edge crossings using barycenter heuristic."""
        _minimize_crossings_iterative(self.layers, self.graph, self.node_layer, self.positions)

    def _assign_coordinates(self, layer_spacing=100, node_spacing=100):
        """Assign x, y coordinates to nodes based on orientation."""
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
        self._assign_coordinates(layer_spacing, node_spacing)

        return self.positions


def _detect_edge_crossing(point_1, point_2, point_3, point_4):
    """
    Detect if two line segments cross using the orientation method.

    Args:
        point_1, point_2: Endpoints of first line segment (x, y tuples)
        point_3, point_4: Endpoints of second line segment (x, y tuples)

    Returns:
        True if segments cross, False otherwise
    """

    def ccw(A, B, C):
        """Check if three points are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Check if segments intersect
    return ccw(point_1, point_3, point_4) != ccw(point_2, point_3, point_4) and ccw(
        point_1, point_2, point_3
    ) != ccw(point_1, point_2, point_4)


def _find_edge_crossings(graph, positions, layers, node_layer):
    """
    Find all edge crossings in the current layout.

    Args:
        graph: NetworkX DiGraph
        positions: Dictionary of node positions
        layers: List of nodes per layer
        node_layer: Dictionary mapping nodes to their layer indices

    Returns:
        List of crossing pairs: [(edge_1, edge_2), ...]
    """
    edges = list(graph.edges())
    crossings = []

    # Compare each pair of edges
    for i, edge_1 in enumerate(edges):
        for edge_2 in edges[i + 1 :]:
            source_1, target_1 = edge_1
            source_2, target_2 = edge_2

            # Skip if edges share a node
            if source_1 in (source_2, target_2) or target_1 in (source_2, target_2):
                continue

            # Skip if edges are in the same or adjacent layers
            if abs(node_layer[source_1] - node_layer[source_2]) <= 1:
                continue

            # Check for crossing
            point_1 = positions.get(source_1)
            point_2 = positions.get(target_1)
            point_3 = positions.get(source_2)
            point_4 = positions.get(target_2)

            if point_1 and point_2 and point_3 and point_4:
                if _detect_edge_crossing(point_1, point_2, point_3, point_4):
                    crossings.append((edge_1, edge_2))

    return crossings


def _resolve_crossings_by_swapping(graph, positions, layers, node_layer):
    """
    Attempt to resolve edge crossings by swapping node positions within layers.

    This function iteratively detects crossings and performs local swaps
    to minimize crossing edges between consecutive layers.

    Args:
        graph: NetworkX DiGraph
        positions: Dictionary of node positions (modified in place)
        layers: List of nodes per layer (modified in place)
        node_layer: Dictionary mapping nodes to their layer indices

    Returns:
        Number of swaps performed
    """
    max_iterations = 10

    for iteration in range(max_iterations):
        crossings = _find_edge_crossings(graph, positions, layers, node_layer)

        # skip if there is no crossing after sorts
        if not crossings:
            break

        swaps_this_iteration = 0

        # Process each crossing
        for edge_1, edge_2 in crossings:
            source_1, target_1 = edge_1
            source_2, target_2 = edge_2

            # Get the layers involved
            layer1_source = node_layer[source_1]
            layer2_source = node_layer[source_2]

            # Determine which layer to perform the swap
            # Try swapping nodes in the layer with sources
            for layer_idx in set([layer1_source, layer2_source]):
                if layer_idx < len(layers) - 1:
                    # Find which nodes in this layer are involved in the crossing
                    nodes_in_layer = layers[layer_idx]

                    # Find other nodes that might benefit from swapping
                    for i, node_a in enumerate(nodes_in_layer):
                        for j, node_b in enumerate(nodes_in_layer):
                            if i >= j or node_a == node_b:
                                continue

                            # Attempt swap
                            layers[layer_idx][i], layers[layer_idx][j] = node_b, node_a

                            # Recalculate positions
                            _recalculate_node_positions_for_each_layer(layer_idx, layers, positions)

                            # Check if crossing is reduced
                            new_crossings = _find_edge_crossings(
                                graph, positions, layers, node_layer
                            )

                            # Keep swap if it reduces crossings
                            if len(new_crossings) < len(crossings):
                                swaps_this_iteration += 1
                                break
                            else:
                                # Revert swap
                                layers[layer_idx][i], layers[layer_idx][j] = node_a, node_b
                                _recalculate_node_positions_for_each_layer(
                                    layer_idx, layers, positions
                                )

                        if swaps_this_iteration > 0:
                            break

        if swaps_this_iteration == 0:
            break


def _recalculate_node_positions_for_each_layer(layer_idx, layers, positions, node_spacing=80):
    """
    Recalculate x,y positions for nodes in a specific layer.

    Args:
        layer_idx: Index of the layer to update
        layers: List of nodes per layer
        positions: Dictionary of positions to update
        layer_spacing: Spacing between layers
        node_spacing: Spacing between nodes in a layer
    """
    nodes = layers[layer_idx]
    num_nodes = len(nodes)

    # Calculate x positions centered
    x_offset = -(num_nodes - 1) * node_spacing / 2

    for i, node in enumerate(nodes):
        x = x_offset + i * node_spacing
        if node in positions:
            # Keep y relative to orientation, update x based on new position
            _, old_y = positions[node]
            positions[node] = (x, old_y)


def _minimize_crossings_iterative(layers, graph, node_layer, positions):
    """
    Enhanced crossing minimization that combines barycenter heuristic with
    iterative swapping to resolve edge crossings.

    Args:
        layers: List of nodes per layer
        graph: NetworkX DiGraph
        node_layer: Dictionary mapping nodes to their layer
        positions: Dictionary of node positions
    """
    # First sorting: apply barycenter heuristic (sort the component by layer based on their
    # predecessors)
    for layer_idx in range(1, len(layers)):
        barycenters = {}

        for node in layers[layer_idx]:
            # "predecessors" is the list that contains the nodes in the immediately higher layer
            # which are directly connected to the current node.

            predecessors = [
                predecessor
                for predecessor in graph.predecessors(node)
                if node_layer.get(predecessor, -1) == layer_idx - 1
            ]

            if predecessors:
                predecessor_positions = [
                    layers[layer_idx - 1].index(predecessor) for predecessor in predecessors
                ]
                # Calculate the center placement among the preprocessors
                barycenters[node] = sum(predecessor_positions) / len(predecessor_positions)
            else:
                # If no predecessor exists in the previous layer, place the node at the end
                barycenters[node] = float("inf")

        # Sort all nodes in the current layer by their barycenter values in ascending order
        layers[layer_idx].sort(key=lambda node: barycenters.get(node, float("inf")))

    # Second sorting: resolve remaining crossings via swapping
    _resolve_crossings_by_swapping(graph, positions, layers, node_layer)

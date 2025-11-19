# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import networkx as nx
from collections import deque


class SugiyamaLayout:
    """
    Implements the Sugiyama hierarchical graph layout algorithm.

    The algorithm has four main phases:
    1. Cycle breaking (ensures DAG property)
    2. Layer assignment (places nodes into layers)
    3. Crossing minimization (reduces edge crossings between layers)
    4. Coordinate assignment (positions nodes within layers)
    """

    def __init__(self, graph, orientation="TB", node_layer_override=None):
        """
        Initialize the Sugiyama layout.

        Args:
            graph: NetworkX DiGraph
            orientation: 'TB' (top-bottom), 'BT' (bottom-top),
                        'LR' (left-right), 'RL' (right-left)
            node_layer_override: Optional dictionary to override layer assignment
        """
        self.graph = graph
        self.orientation = orientation
        self.layers = []
        self.node_layer = {}
        self.positions = {}
        self.node_layer_override = node_layer_override

    def _break_cycles(self):
        """Remove cycles by reversing edges in a feedback arc set."""
        if nx.is_directed_acyclic_graph(self.graph):
            return set()

        reversed_edges = set()
        G_copy = self.graph.copy()

        while not nx.is_directed_acyclic_graph(G_copy):
            max_node = None
            max_diff = float("-inf")

            for node in G_copy.nodes():
                diff = G_copy.out_degree(node) - G_copy.in_degree(node)
                if diff > max_diff:
                    max_diff = diff
                    max_node = node

            if max_node is None:
                break

            incoming = list(G_copy.in_edges(max_node))
            for source, target in incoming:
                G_copy.remove_edge(source, target)
                G_copy.add_edge(target, source)
                reversed_edges.add((source, target))

        return reversed_edges

    def _assign_layers(self):
        """Assign each node to a layer using provided override or longest path from roots."""
        # Use override if provided
        if self.node_layer_override:
            # Convert float values to integers for layer assignment
            self.node_layer = {node: int(layer) for node, layer in self.node_layer_override.items()}
        else:
            # Fallback to longest path method
            roots = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

            if not roots:
                try:
                    roots = [list(nx.topological_sort(self.graph))[0]]
                except nx.NetworkXUnfeasible:
                    # Fallback when graph is not a DAG
                    roots = [list(self.graph.nodes())[0]]

            self.node_layer = {node: 0 for node in self.graph.nodes()}
            queue = deque([(node, 0) for node in roots])
            visited = set()

            while queue:
                node, layer = queue.popleft()

                if node in visited:
                    continue
                visited.add(node)

                self.node_layer[node] = max(self.node_layer.get(node, 0), layer)

                for successor in self.graph.successors(node):
                    successor_layer = self.node_layer[node] + 1
                    if successor_layer > self.node_layer.get(successor, 0):
                        queue.append((successor, successor_layer))

        max_layer = max(self.node_layer.values()) if self.node_layer else 0
        self.layers = [[] for _ in range(max_layer + 1)]

        for node, layer in self.node_layer.items():
            self.layers[layer].append(node)

        return self.layers

    def _minimize_crossings(self):
        """Reorder nodes within layers to minimize edge crossings using barycenter heuristic."""
        _enhanced_minimize_crossings(self.layers, self.graph, self.node_layer, self.positions)

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
        self._break_cycles()
        self._assign_layers()
        self._minimize_crossings()
        self._assign_coordinates(layer_spacing, node_spacing)

        return self.positions


def _detect_edge_crossing(p1, p2, p3, p4):
    """
    Detect if two line segments cross using the orientation method.

    Args:
        p1, p2: Endpoints of first line segment (x, y tuples)
        p3, p4: Endpoints of second line segment (x, y tuples)

    Returns:
        True if segments cross, False otherwise
    """

    def ccw(A, B, C):
        """Check if three points are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Check if segments intersect
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def _find_edge_crossings(graph, positions, layers, node_layer):
    """
    Find all edge crossings in the current layout.

    Args:
        graph: NetworkX DiGraph
        positions: Dictionary of node positions
        layers: List of nodes per layer
        node_layer: Dictionary mapping nodes to their layer indices

    Returns:
        List of crossing pairs: [(edge1, edge2), ...]
    """
    edges = list(graph.edges())
    crossings = []

    # Compare each pair of edges
    for i, edge1 in enumerate(edges):
        for edge2 in edges[i + 1 :]:
            src1, tgt1 = edge1
            src2, tgt2 = edge2

            # Skip if edges share a node
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue

            # Skip if edges are in the same or adjacent layers
            if abs(node_layer[src1] - node_layer[src2]) <= 1:
                continue

            # Check for crossing
            p1 = positions.get(src1)
            p2 = positions.get(tgt1)
            p3 = positions.get(src2)
            p4 = positions.get(tgt2)

            if p1 and p2 and p3 and p4:
                if _detect_edge_crossing(p1, p2, p3, p4):
                    crossings.append((edge1, edge2))

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
    swap_count = 0

    for iteration in range(max_iterations):
        crossings = _find_edge_crossings(graph, positions, layers, node_layer)

        if not crossings:
            break

        swaps_this_iteration = 0

        # Process each crossing
        for edge1, edge2 in crossings:
            src1, tgt1 = edge1
            src2, tgt2 = edge2

            # Get the layers involved
            layer1_src = node_layer[src1]
            layer2_src = node_layer[src2]

            # Determine which layer to perform the swap
            # Try swapping nodes in the layer with sources
            for layer_idx in set([layer1_src, layer2_src]):
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
                            _recalculate_positions_for_layer(layer_idx, layers, positions)

                            # Check if crossing is reduced
                            new_crossings = _find_edge_crossings(
                                graph, positions, layers, node_layer
                            )

                            # Keep swap if it reduces crossings
                            if len(new_crossings) < len(crossings):
                                swaps_this_iteration += 1
                                swap_count += 1
                                break
                            else:
                                # Revert swap
                                layers[layer_idx][i], layers[layer_idx][j] = node_a, node_b
                                _recalculate_positions_for_layer(layer_idx, layers, positions)

                        if swaps_this_iteration > 0:
                            break

        if swaps_this_iteration == 0:
            break

    return swap_count


def _recalculate_positions_for_layer(
    layer_idx, layers, positions, layer_spacing=100, node_spacing=80
):
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
            old_x, old_y = positions[node]
            positions[node] = (x, old_y)


def _enhanced_minimize_crossings(layers, graph, node_layer, positions):
    """
    Enhanced crossing minimization that combines barycenter heuristic with
    iterative swapping to resolve edge crossings.

    Args:
        layers: List of nodes per layer
        graph: NetworkX DiGraph
        node_layer: Dictionary mapping nodes to their layer
        positions: Dictionary of node positions
    """
    # First pass: apply barycenter heuristic
    for layer_idx in range(1, len(layers)):
        barycenters = {}

        for node in layers[layer_idx]:
            predecessors = [
                p for p in graph.predecessors(node) if node_layer.get(p, -1) == layer_idx - 1
            ]

            if predecessors:
                pred_positions = [layers[layer_idx - 1].index(p) for p in predecessors]
                barycenters[node] = sum(pred_positions) / len(pred_positions)
            else:
                barycenters[node] = float("inf")

        layers[layer_idx].sort(key=lambda n: barycenters.get(n, float("inf")))

    # Second pass: resolve remaining crossings via swapping
    swap_count = _resolve_crossings_by_swapping(graph, positions, layers, node_layer)

    return swap_count

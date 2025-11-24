.. _sort_mechanism:

=================
Hierarchy sorting
=================
The sorting mechanism inspired by the `Tutte's drawing algorithm <https://www.youtube.com/watch?v=mEzPPMhR8XE&t=45s>`_
is utilized before each visualization in a `static html <https://www.geeksforgeeks.org/computer-networks/difference-between-static-and-dynamic-web-pages/>`_
format. This prevents connection crossing between two layers and allows a more organized presentation of the powertrain
architecture.

Tutte's drawing algorithm
=========================
The Tutte spring theorem, proved by W. T. Tutte :cite:`tutte:1963` in 1963, provides an algorithmic approach with unique
solution for drawing 3-connected planar graphs. The following garphe is an example of a 3-connected planar graphs.

.. image:: ../../../img/tuttes.svg
    :width: 600
    :align: center

This method has since been widely applied in computing embeddings of complex planar polygons when the boundary nodes are
appropriately selected.

Reference level
===============
The reference level definition is one of the two preparation steps before the actual sorting. Two approaches are
considered to define the root of the powertrain architecture tree, from-storage and from-propulsor. The from-storage
approach (default) sets all the energy storage components into the base reference level of the whole architecture. The
highlighted layer represented the base of the hierarchy.

.. image:: ../../../img/from_storage.svg
    :width: 600
    :align: center

Similar as the from-storage approach, the from-propulsor approach sets the propulsor component as the top reference
level for the whole architecture.

.. image:: ../../../img/from_propulsor.svg
    :width: 600
    :align: center

Boundary nodes
==============
The boundary nodes definition is a critical step for the boundaries of the Tutte's drawing that ensures non-trivial
solution from the linear system. The boundary nodes are consisted with the nodes in the reference layer and the farthest
layer from the reference layer.

.. image:: ../../../img/boundary_node.svg
    :width: 600
    :align: center


Linear system solving
=====================
The fundamental principle of Tutte's drawing algorithm is that each interior node must be positioned at the average
(barycenter) of its neighbors' coordinates. This property ensures that edges connecting a node to its neighbors have
minimal tension and naturally reduces edge crossings. To applies in a generally, some rearrangements of the
equations are required. The barycenter of a polygon can be expressed as:

.. math::
    x_i = \frac{\sum_{j \in N(i)} x_j}{|N(i)|} \quad \text{and} \quad y_i = \frac{\sum_{j \in N(i)} y_j}{|N(i)|}


Rearrange the barycenter constraint to standard linear form and applied to all the neighboring nodes:

.. math::

    |N(i)| \cdot x_i - \sum_{j \in N(i)} x_j = 0 \\
    |N(i)| \cdot y_i - \sum_{j \in N(i)} y_j = 0

where :math:`N(i)` is the set of neighbors of node :math:`i` and :math:`|N(i)|` is the degree (number of neighbors).

Then separate interior nodes (unknowns) from boundary nodes (known) and rewrite into matrices:

.. math::
    \begin{cases}
        \left( |N(i)| \cdot x_i - \sum_{j \in \text{interior}} x_j \right)
        = \sum_{j \in \text{boundary}} x_j \\[8pt]
        \left( |N(i)| \cdot y_i - \sum_{j \in \text{interior}} y_j \right)
        = \sum_{j \in \text{boundary}} y_j
    \end{cases}
    \;\Rightarrow\;
    \begin{cases}
        L \mathbf{x} = \mathbf{b}_x \\
        L \mathbf{y} = \mathbf{b}_y
    \end{cases}

The Laplacian matrix :math:`L` is defined as:

.. math::

    L_{ii} = |N(i)|,\qquad
        L_{ij} =
        \begin{cases}
            -1, & \text{if interior node } i \text{ and } j \text{ are connected},\\[6pt]
             0, & \text{otherwise}.
        \end{cases}

Finally, solve the linear system to obtain the coordinates

.. math::
    \mathbf{x}_{\text{interior}} = L^{-1} \mathbf{b}_x \\
    \mathbf{y}_{\text{interior}} = L^{-1} \mathbf{b}_y


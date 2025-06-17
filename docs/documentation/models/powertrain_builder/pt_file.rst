.. _pt-file:

=============================
Powertrain configuration file
=============================

The powertrain configuration file (PT file) defines the powertrain components and their connections. It consists of
four sections:

.. code:: yaml

    title:
    #  Similar to the model configuration file, this gives a name to the architecture.

    power_train_components:
    # Lists all components, including their names, IDs, and option specifications.

    component_connections:
    # Describes how components are connected.

    watcher_file_path:
    # Specifies the path to the CSV file that contains powertrain performance at each time step.

*********************
Powertrain components
*********************

All the components required for the powertrain need to be specified at this section with the following format.
The `id` of all existed component can be found in :ref:`Component Constraints and ID <constraint-id>`.

Typical component
=================
This format is mostly applied to all the powertrain components exist in FAST-GA-HE, except several connection
components allowing multiple inputs or outputs connections.

.. code:: yaml

    component_1:
    id: fastga_he.pt_component.<component type>
    options:
        # Other options of the component if applicable
    position: # Installation position

Multiple connection component
=============================
Here demonstrates how the multiple connection components are defined in the PT files.

.. code:: yaml

    fuel_system_1:
    id: fastga_he.pt_component.fuel_system
    options:
      number_of_engines: # Number of engines to connect
      number_of_tanks: # Number of tanks to connect
    position:  # Installation position

    h2_fuel_system_1:
    id: fastga_he.pt_component.h2_fuel_system
    options:
      number_of_power_sources: # Number of power sources to connect
      number_of_tanks: # Number of tanks to connect
    position:  # Installation position

    dc_bus_1:
    id: fastga_he.pt_component.dc_bus
    options:
      number_of_inputs: # Number of inputs to connect
      number_of_outputs: # Number of outputs to connect
    position:  # Installation position

For the DC splitter and the planetary gear component, the working logic of allowed connection options are explained with
this diagram :cite:`lutz:2025`.

.. image:: ../../../img/splitter.svg
    :width: 800
    :align: center

.. code:: yaml

    dc_splitter_1:
    id: fastga_he.pt_component.dc_splitter
    options:
      splitter_mode: # percent_split by default or power_share
    position: # Installation position

    planetary_gear_1:
    id: fastga_he.pt_component.planetary_gear
    options:
      gear_mode: # percent_split by default or power_share
    position: # Installation position


*********************
Component connections
*********************
This section defines the component sequence and the connections of the powertrain architecture. For each connection, the
component placed at the source of the connection is the input value provider and the component placed at target for a
connection is receiver of those values.

One-to-one connection
=====================
This format is applied to all typical powertrain components with an one-to-one relation.

.. code:: yaml

    - source: component_1
      target: component_2


Multiple input / output connection
==================================
When having connection with multiple components, the connection index must be specified. The number of connections
must match the number defined in the ``power_train_components`` section. The ``<index of connection>`` should be an
integer starting from 1 up to the number specified in ``power_train_components``.

.. code:: yaml

    # If the multiple connection component is the source of this connection
    - source: [component_1, <index of connection>]
      target: component_2

    # If the multiple connection component is the target of this connection
    - source: component_1
      target: [component_2, <index of connection>]
.. _templates:

=================
PT file templates
=================
This section presents several powertrain designs using the powertrain configuration file format.

Single ICE architecture
***********************

.. code:: yaml

    title: Single ICE powertrain

    power_train_components:
      propeller_1:
        id: fastga_he.pt_component.propeller
        position: in_the_nose
      ice_1:
        id: fastga_he.pt_component.internal_combustion_engine
        position: in_the_front
      fuel_system_1:
        id: fastga_he.pt_component.fuel_system
        options:
          number_of_engines: 1
          number_of_tanks: 2
        position: in_the_front
      fuel_tank_1:
        id: fastga_he.pt_component.fuel_tank
        position: inside_the_wing
      fuel_tank_2:
        id: fastga_he.pt_component.fuel_tank
        position: inside_the_wing
        symmetrical: fuel_tank_1

    component_connections:
      - source: propeller_1
        target: ice_1

      - source: ice_1
        target: [fuel_system_1, 1]

      - source: [fuel_system_1, 1]
        target: fuel_tank_1

      - source: [fuel_system_1, 2]
        target: fuel_tank_2

    watcher_file_path: ../results/pt_watcher.csv

Dual turboshaft architecture
****************************

.. code:: yaml

    title: Dual turboshaft powertrain

    power_train_components:
      propeller_1:
        id: fastga_he.pt_component.propeller
        position: on_the_wing
      turboshaft_1:
        id: fastga_he.pt_component.turboshaft
        position: on_the_wing
      propeller_2:
        id: fastga_he.pt_component.propeller
        position: on_the_wing
      turboshaft_2:
        id: fastga_he.pt_component.turboshaft
        position: on_the_wing
      fuel_system_1:
        id: fastga_he.pt_component.fuel_system
        options:
          number_of_engines: 2
          number_of_tanks: 2
        position: in_the_wing
      fuel_tank_1:
        id: fastga_he.pt_component.fuel_tank
        position: inside_the_wing
      fuel_tank_2:
        id: fastga_he.pt_component.fuel_tank
        position: inside_the_wing
        symmetrical: fuel_tank_1

    component_connections:
      - source: propeller_1
        target: turboshaft_1

      - source: propeller_2
        target: turboshaft_2

      - source: turboshaft_1
        target: [fuel_system_1, 1]

      - source: turboshaft_2
        target: [ fuel_system_1, 2]

      - source: [fuel_system_1, 1]
        target: fuel_tank_1

      - source: [fuel_system_1, 2]
        target: fuel_tank_2

    watcher_file_path: ../results/pt_watcher.csv

Single PMSM electric architecture
*********************************

.. code:: yaml

    title: Single PMSM electric powertrain

    power_train_components:
      propeller_1:
        id: fastga_he.pt_component.propeller
        position: in_the_nose
      motor_1:
        id: fastga_he.pt_component.pmsm
        position: in_the_nose
      inverter_1:
        id: fastga_he.pt_component.inverter
        position: in_the_front
      dc_bus_1:
        id: fastga_he.pt_component.dc_bus
        options:
          number_of_inputs: 1
          number_of_outputs: 1
        position: in_the_front
      harness_1:
        id: fastga_he.pt_component.dc_line
        position: from_rear_to_front
      dc_splitter_1:
        id: fastga_he.pt_component.dc_splitter
        position: in_the_back
      dc_sspc_1:
        id: fastga_he.pt_component.dc_sspc
        options:
          closed_by_default: True
        position: in_the_front
      battery_pack_1:
        id: fastga_he.pt_component.battery_pack
        position: in_the_front
      dc_sspc_2:
        id: fastga_he.pt_component.dc_sspc
        options:
          closed_by_default: True
        position: in_the_back
      battery_pack_2:
        id: fastga_he.pt_component.battery_pack
        position: in_the_back

    component_connections:
      - source: propeller_1
        target: motor_1

      - source: motor_1
        target: inverter_1

      - source: inverter_1
        target: [dc_bus_1, 1]

      - source: [dc_bus_1, 1]
        target: harness_1

      - source: harness_1
        target: dc_splitter_1

      - source: [dc_splitter_1, 1]
        target: dc_sspc_1

      - source: dc_sspc_1
        target: battery_pack_1

      - source: [dc_splitter_1, 2 ]
        target: dc_sspc_2

      - source: dc_sspc_2
        target: battery_pack_2

    watcher_file_path: ../results/pt_watcher.csv

Dual PMSM single propeller architecture
***************************************

.. code:: yaml

    title: Dual PMSM single propeller powertrain

    power_train_components:
      propeller_1:
        id: fastga_he.pt_component.propeller
      planetary_gear_1:
        id: fastga_he.pt_component.planetary_gear
        options:
          gear_mode: power_share
      motor_1:
        id: fastga_he.pt_component.pmsm
      motor_2:
        id: fastga_he.pt_component.pmsm
      inverter_1:
        id: fastga_he.pt_component.inverter
      inverter_2:
        id: fastga_he.pt_component.inverter
      dc_bus_1:
        id: fastga_he.pt_component.dc_bus
        options:
          number_of_inputs: 1
          number_of_outputs: 1
      dc_bus_2:
        id: fastga_he.pt_component.dc_bus
        options:
          number_of_inputs: 1
          number_of_outputs: 1
      harness_1:
        id: fastga_he.pt_component.dc_line
      harness_2:
        id: fastga_he.pt_component.dc_line
      dc_bus_5:
        id: fastga_he.pt_component.dc_bus
        options:
          number_of_inputs: 1
          number_of_outputs: 2
      dc_dc_converter_1:
        id: fastga_he.pt_component.dc_dc_converter
      battery_pack_1:
        id: fastga_he.pt_component.battery_pack

    component_connections:
      - source: propeller_1
        target: [planetary_gear_1, 1]

      - source: [planetary_gear_1, 1]
        target: motor_1

      - source: [ planetary_gear_1, 2 ]
        target: motor_2

      - source: motor_1
        target: inverter_1

      - source: motor_2
        target: inverter_2

      - source: inverter_1
        target: [dc_bus_1, 1]

      - source: inverter_2
        target: [dc_bus_2, 1]

      - source: [dc_bus_1, 1]
        target: harness_1

      - source: [dc_bus_2, 1]
        target: harness_2

      - source: harness_1
        target: [dc_bus_5, 1]
      - source: harness_2
        target: [dc_bus_5, 2]

      - source: [dc_bus_5, 1]
        target: dc_dc_converter_1

      - source: dc_dc_converter_1
        target: battery_pack_1

    watcher_file_path: ../results/pt_watcher.csv
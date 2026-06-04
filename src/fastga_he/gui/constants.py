# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

POSSIBLE_POSITIONS = {
    "DC_bus": ["inside_the_wing", "in_the_front", "in_the_back"],
    "DC_cable_harness": [],
    "DC_SSPC": ["inside_the_wing", "in_the_front", "in_the_back"],
    "DC_splitter": ["inside_the_wing", "in_the_front", "in_the_back"],
    "rectifier": ["inside_the_wing", "in_the_front", "in_the_back"],
    "DC_DC_converter": ["inside_the_wing", "in_the_front", "in_the_back"],
    "inverter": ["inside_the_wing", "in_the_front", "in_the_back"],
    "speed_reducer": ["inside_the_wing", "in_the_front", "in_the_back"],
    "planetary_gear": ["inside_the_wing", "in_the_front", "in_the_back"],
    "gearbox": ["inside_the_wing", "in_the_front", "in_the_back"],
    "PMSM": ["on_the_wing", "in_the_nose"],
    "SM_PMSM": ["on_the_wing", "in_the_nose"],
    "aux_load": ["inside_the_wing", "in_the_front", "in_the_back"],
    "battery_pack": ["inside_the_wing", "wing_pod", "in_the_front", "in_the_back", "underbelly"],
    "generator": ["inside_the_wing", "in_the_front", "in_the_back"],
    "turbo_generator": ["inside_the_wing", "in_the_front", "in_the_back"],
    "ICE": ["on_the_wing", "in_the_front", "in_the_back"],
    "high_rpm_ICE": ["on_the_wing", "in_the_front", "in_the_back"],
    "turboshaft": ["on_the_wing", "in_the_front", "in_the_back"],
    "PEMFC_stack": ["in_the_front", "wing_pod", "underbelly", "in_the_back"],
    "propeller": ["on_the_wing", "in_the_nose"],
    "fuel_tank": ["inside_the_wing", "wing_pod", "in_the_fuselage"],
    "gaseous_hydrogen_tank": ["in_the_cabin", "wing_pod", "in_the_back", "underbelly"],
    "fuel_system": ["in_the_wing", "in_the_front", "in_the_back"],
    "H2_fuel_system": ["in_the_front", "in_the_middle", "in_the_rear"],
}

ICON_TYPE = {
    "connector": [
        "bus_bar",
        "cable",
        "switch",
        "splitter",
        "rectifier",
        "dc_converter",
        "inverter",
        "gearbox",
        "fuel_system",
    ],
    "load": ["e_motor", "dc_load"],
    "source": [
        "battery",
        "generator",
        "ice",
        "turbine",
        "fuel_cell",
    ],
    "propulsor": ["propeller"],
    "tank": ["fuel_tank"],
}
# Dict of component_type -> list of component_keys (e.g. "source" -> ["battery", "generator", …])

POSSIBLE_COMPONENT_TYPES = {
    "bus_bar": "DC_bus",
    "cable": "DC_cable_harness",
    "switch": "DC_SSPC",
    "splitter": "DC_splitter",
    "rectifier": "rectifier",
    "dc_converter": "DC_DC_converter",
    "inverter": "inverter",
    "gearbox": ["speed_reducer", "planetary_gear", "gearbox"],
    "e_motor": ["PMSM", "SM_PMSM"],
    "dc_load": "aux_load",
    "battery": "battery_pack",
    "generator": ["generator", "turbo_generator"],
    "ice": ["ICE", "high_rpm_ICE"],
    "turbine": "turboshaft",
    "fuel_cell": "PEMFC_stack",
    "propeller": "propeller",
    "fuel_tank": ["fuel_tank", "gaseous_hydrogen_tank"],
    "fuel_system": ["fuel_system", "H2_fuel_system"],
}

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

ID = "id"
# CN is used to translate the ID to the prefix used in the OpenMDAO names for the components,
# e.g: PerformancesPropeller = Performances + Propeller
CN = "OM_components_name"
# CN_ID is used to store the id of the components in question, "battery_pack_id", "propeller_id"
CN_ID = "OM_components_id"
# CT is used to store the type of the components in question, "battery_pack", "propeller"
CT = "components_type"
# CTC is used to store the class of the components type type, like "propulsor", "connector", ...
CTC = "components_type_class"
ATT = "attributes"
# The IN and OUT field contain the input and output in the system sense of the term,
# meaning the output of the prop is the propulsive power while its input is the mechanical power.
# In those inputs, there will be tuple of two element: the first filled when the system
# input/output is an openmdao input the other if the opposite
IN = "inputs"
OUT = "outputs"
# The PT field contains the variables that must be promoted from aircraft level for the component
# to work
PT = "promoted_variables"
# The MP field will contain the variables that will be of interest for the analysis of the
# mission performances and that will be registered in the power train performances CSV file
MP = "mission_performances_watcher"
# ICON contains the name of the icon used to represent this component when displaying the graph
ICON = "icon_for_network_graph"
# ICON_SIZE contains the size of th icon used to represent this component when displaying the graph
ICON_SIZE = "icon_size_for_network_graph"
# RSD contains the list of residuals that will be displayed when using the residuals viewer
RSD = "interesting_residuals"
# SETS_V contains a boolean that indicates if, as part of the load flow analysis method,
# this component set the voltage of one part of the architecture
SETS_V = "set_voltage"
# IO_INDEP_V contains a boolean that indicates if the voltage at the output of a component is
# independent of the voltage at the input, which would separate two part of the architecture in
# terms of voltage. For instance, regardless of the voltage at the input of a converter,
# its output will always be at target_voltage_out, only current will change.
IO_INDEP_V = "input_output_voltage_independant"
# V_TO_SET contains a list, for each type of components of their voltage caracteristic that can
# be set
V_TO_SET = "voltage_to_precondition"

PROPELLER = {
    ID: "fastga_he.pt_component.propeller",
    CN: "Propeller",
    CN_ID: "propeller_id",
    CT: "propeller",
    ATT: None,
    PT: ["true_airspeed", "altitude"],
    IN: [(None, "rpm"), (None, "shaft_power_in")],
    OUT: None,
    CTC: "propulsor",
    MP: [
        {"rpm": "1/min"},
        {"shaft_power_in": "kW"},
        {"torque_in": "N*m"},
        {"efficiency": None},
        {"advance_ratio": None},
        {"reynolds_D": None},
        {"tip_mach": None},
        {"thrust_coefficient": None},
        {"power_coefficient": None},
    ],
    ICON: "propeller",
    ICON_SIZE: 30,
    RSD: ["thrust_coefficient", "power_coefficient"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
}
PMSM = {
    ID: "fastga_he.pt_component.pmsm",
    CN: "PMSM",
    CN_ID: "motor_id",
    CT: "PMSM",
    ATT: None,
    PT: ["settings:*"],
    IN: [
        (None, "ac_current_rms_in_one_phase"),
        (None, "ac_voltage_peak_in"),
        (None, "ac_voltage_rms_in"),
    ],
    OUT: [("rpm", None), ("shaft_power_out", None)],
    CTC: "propulsive_load",
    MP: [
        {"efficiency": None},
        {"torque_out": "N*m"},
        {"ac_current_rms_in": "A"},
        {"ac_voltage_rms_in": "V"},
    ],
    ICON: "e_motor",
    ICON_SIZE: 40,
    RSD: ["ac_current_rms_in", "ac_voltage_rms_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["ac_voltage_rms_in", "ac_voltage_peak_in"],
}
INVERTER = {
    ID: "fastga_he.pt_component.inverter",
    CN: "Inverter",
    CN_ID: "inverter_id",
    CT: "inverter",
    ATT: None,
    PT: ["settings:*"],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [
        ("ac_current_rms_out_one_phase", None),
        ("ac_voltage_peak_out", None),
        ("ac_voltage_rms_out", None),
    ],
    CTC: "connector",
    MP: [
        {"modulation_index": None},
        {"efficiency": None},
        {"dc_current_in": "A"},
        {"diode_temperature": "degK"},
        {"IGBT_temperature": "degK"},
        {"casing_temperature": "degK"},
        {"losses_inverter": "W"},
        {"switching_losses_IGBT": "W"},
        {"switching_losses_diode": "W"},
        {"conduction_losses_IGBT": "W"},
        {"conduction_losses_diode": "W"},
    ],
    ICON: "inverter",
    ICON_SIZE: 30,
    RSD: ["modulation_index", "dc_current_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
}
DC_BUS = {
    ID: "fastga_he.pt_component.dc_bus",
    CN: "DCBus",
    CN_ID: "dc_bus_id",
    CT: "DC_bus",
    ATT: ["number_of_inputs", "number_of_outputs"],
    PT: [],
    IN: [(None, "dc_voltage"), ("dc_current_in_", None)],
    OUT: [(None, "dc_voltage"), ("dc_current_out_", None)],
    CTC: "connector",
    MP: [{"dc_voltage": "V"}],
    ICON: "bus_bar",
    ICON_SIZE: 20,
    RSD: ["dc_voltage"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["dc_voltage"],
}
DC_LINE = {
    ID: "fastga_he.pt_component.dc_line",
    CN: "Harness",
    CN_ID: "harness_id",
    CT: "DC_cable_harness",
    ATT: None,
    PT: ["exterior_temperature", "settings:*", "time_step"],
    IN: [("dc_voltage_in", None), (None, "dc_current")],
    OUT: [("dc_voltage_out", None), (None, "dc_current")],
    CTC: "connector",
    MP: [{"dc_current": "A"}, {"cable_temperature": "degK"}, {"conduction_losses": "W"}],
    ICON: "cable",
    ICON_SIZE: 30,
    RSD: ["dc_current"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
}
DC_DC_CONVERTER = {
    ID: "fastga_he.pt_component.dc_dc_converter",
    CN: "DCDCConverter",
    CN_ID: "dc_dc_converter_id",
    CT: "DC_DC_converter",
    ATT: None,
    PT: [],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [("dc_voltage_out", None), (None, "dc_current_out")],
    CTC: "connector",
    MP: [
        {"efficiency": None},
        {"duty_cycle": None},
        {"dc_current_in": "A"},
        {"dc_current_out": "A"},
    ],
    ICON: "dc_converter",
    ICON_SIZE: 30,
    RSD: ["dc_current_in", "dc_current_out", "duty_cycle"],
    SETS_V: True,
    IO_INDEP_V: True,
    V_TO_SET: [],  # It is a bit paradoxical but you cant set a setter's voltage :p
}
BATTERY_PACK = {
    ID: "fastga_he.pt_component.battery_pack",
    CN: "BatteryPack",
    CN_ID: "battery_pack_id",
    CT: "battery_pack",
    ATT: None,
    PT: ["time_step"],
    IN: None,
    OUT: [(None, "voltage_out"), ("dc_current_out", None)],
    CTC: "source",
    MP: [
        {"c_rate": "1/h"},
        {"state_of_charge": "percent"},
        {"open_circuit_voltage": "V"},
        {"internal_resistance": "ohm"},
        {"voltage_out": "V"},
        {"efficiency": None},
        {"relative_capacity": None},
    ],
    ICON: "battery",
    ICON_SIZE: 40,
    RSD: ["voltage_out", "state_of_charge", "c_rate"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
}
DC_SSPC = {
    ID: "fastga_he.pt_component.dc_sspc",
    CN: "DCSSPC",
    CN_ID: "dc_sspc_id",
    CT: "DC_SSPC",
    ATT: ["closed_by_default"],
    PT: [],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [(None, "dc_voltage_out"), ("dc_current_out", None)],
    CTC: "connector",
    MP: [
        {"dc_current_in": "A"},
        {"efficiency": None},
        {"power_losses": "W"},
        {"dc_voltage_out": "V"},
    ],
    ICON: "switch",
    ICON_SIZE: 2,
    RSD: ["dc_voltage_out", "dc_current_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["dc_voltage_out"],
}
DC_SPLITTER = {
    ID: "fastga_he.pt_component.dc_splitter",
    CN: "DCSplitter",
    CN_ID: "dc_splitter_id",
    CT: "DC_splitter",
    ATT: ["splitter_mode"],
    PT: [],
    IN: [(None, "dc_voltage_in_"), ("dc_current_in_", None)],
    OUT: [(None, "dc_voltage"), ("dc_current_out", None)],
    CTC: "connector",
    MP: [{"dc_voltage": "V"}, {"power_split": "percent"}],
    ICON: "splitter",
    ICON_SIZE: 20,
    RSD: ["power_split", "dc_voltage"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["dc_voltage", "dc_voltage_in_1", "dc_voltage_in_2"],
}
RECTIFIER = {
    ID: "fastga_he.pt_component.rectifier",
    CN: "Rectifier",
    CN_ID: "rectifier_id",
    CT: "rectifier",
    ATT: None,
    PT: [],
    IN: [
        (None, "ac_current_rms_in_one_phase"),
        ("ac_voltage_rms_in", None),
        ("ac_voltage_peak_in", None),
    ],
    OUT: [("dc_voltage_out", None), (None, "dc_current_out")],
    CTC: "connector",
    MP: [
        {"efficiency": None},
        {"modulation_index": None},
        {"ac_current_rms_in_one_phase": "A"},
        {"dc_current_out": "A"},
    ],
    ICON: "rectifier",
    ICON_SIZE: 30,
    RSD: ["dc_current_out", "modulation_index", "ac_current_rms_in_one_phase"],
    SETS_V: True,
    IO_INDEP_V: True,
    V_TO_SET: [],
}
GENERATOR = {
    ID: "fastga_he.pt_component.generator",
    CN: "Generator",
    CN_ID: "generator_id",
    CT: "generator",
    ATT: None,
    PT: [],
    IN: [(None, "rpm"), (None, "shaft_power_in")],
    OUT: [
        ("ac_current_rms_out_one_phase", None),
        (None, "ac_voltage_rms_out"),
        (None, "ac_voltage_peak_out"),
    ],
    CTC: "connector",
    MP: [
        {"rpm": "1/min"},
        {"shaft_power_in": "kW"},
        {"power_losses": "W"},
        {"torque_in": "N*m"},
        {"efficiency": None},
        {"ac_voltage_rms_out": "V"},
        {"ac_voltage_peak_out": "V"},
    ],
    ICON: "e_motor",
    ICON_SIZE: 30,
    RSD: ["shaft_power_in", "ac_voltage_rms_out", "efficiency"],
    SETS_V: True,
    IO_INDEP_V: False,
    V_TO_SET: ["ac_voltage_rms_out", "ac_voltage_peak_out"],
}
ICE = {
    ID: "fastga_he.pt_component.internal_combustion_engine",
    CN: "ICE",
    CN_ID: "ice_id",
    CT: "ICE",
    ATT: None,
    PT: ["time_step", "altitude", "settings:*"],
    IN: None,
    OUT: [("rpm", None), ("shaft_power_out", None)],
    CTC: ["source", "propulsive_load"],
    MP: [
        {"torque_out": "N*m"},
        {"specific_fuel_consumption": "kg/kW/h"},
        {"mean_effective_pressure": "bar"},
        {"fuel_consumption": "kg/h"},
        {"fuel_consumed_t": "kg"},
    ],
    ICON: "ice",
    ICON_SIZE: 40,
    RSD: ["fuel_consumption", "mean_effective_pressure", "torque_out"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
}

KNOWN_COMPONENTS = [
    PROPELLER,
    PMSM,
    INVERTER,
    DC_BUS,
    DC_LINE,
    DC_DC_CONVERTER,
    BATTERY_PACK,
    DC_SSPC,
    DC_SPLITTER,
    RECTIFIER,
    GENERATOR,
    ICE,
]

KNOWN_ID = []

DICTIONARY_CN = {}
DICTIONARY_CN_ID = {}
DICTIONARY_CT = {}
DICTIONARY_ATT = {}
DICTIONARY_PT = {}
DICTIONARY_IN = {}
DICTIONARY_OUT = {}
DICTIONARY_CTC = {}
DICTIONARY_MP = {}
DICTIONARY_ICON = {}
DICTIONARY_ICON_SIZE = {}
DICTIONARY_RSD = {}
DICTIONARY_SETS_V = {}
DICTIONARY_IO_INDEP_V = {}
DICTIONARY_V_TO_SET = {}

for known_component in KNOWN_COMPONENTS:
    KNOWN_ID.append(known_component[ID])
    DICTIONARY_CN[known_component[ID]] = known_component[CN]
    DICTIONARY_CN_ID[known_component[ID]] = known_component[CN_ID]
    DICTIONARY_CT[known_component[ID]] = known_component[CT]
    DICTIONARY_ATT[known_component[ID]] = known_component[ATT]
    DICTIONARY_PT[known_component[ID]] = known_component[PT]
    DICTIONARY_IN[known_component[ID]] = known_component[IN]
    DICTIONARY_OUT[known_component[ID]] = known_component[OUT]
    DICTIONARY_CTC[known_component[ID]] = known_component[CTC]
    DICTIONARY_MP[known_component[ID]] = known_component[MP]
    DICTIONARY_ICON[known_component[ID]] = known_component[ICON]
    DICTIONARY_ICON_SIZE[known_component[ID]] = known_component[ICON_SIZE]
    DICTIONARY_RSD[known_component[ID]] = known_component[RSD]
    DICTIONARY_SETS_V[known_component[ID]] = known_component[SETS_V]
    DICTIONARY_IO_INDEP_V[known_component[ID]] = known_component[IO_INDEP_V]
    DICTIONARY_V_TO_SET[known_component[ID]] = known_component[V_TO_SET]

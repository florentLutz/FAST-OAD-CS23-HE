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
# The PT field contains the variables that must be promoted from aircraft level for the components
# performances group to work
PT = "promoted_variables"
# The SPT field contains the variables that must be promoted from aircraft level for the components
# slipstream group to work
SPT = "slipstream_promoted_variables"
# The PTS field contains the variables that must be connected from the performances computation
# of a component to the computation of the slipstream effect caused by that component.
PTS = "performances_to_slipstream_connection"
# The MP field will contain the variables that will be of interest for the analysis of the
# mission performances and that will be registered in the power train performances CSV file
MP = "mission_performances_watcher"
# The MP field will contain the variables that will be of interest for the analysis of the
# slipstream effects and that will be registered in the power train performances CSV file
SMP = "slipstream_mission_performances_watcher"
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
# P_TO_SET contains a list of tuple with for each type of components of their power characteristic
# that can be set and whether they are at the input of the component ("in") or at the output (
# "out"). Since I did not fully do the naming correctly, the name of the OpenMDAO variable which
# contains the power cannot tell me whether it is at the input or output of the component (
# "active_power" for the PMSM for instance)
P_TO_SET = "power_to_precondition"
# I_TO_SET contains a list of tuple with for each type of components of their current
# characteristic that can be set and whether they are at the input of the component ("in") or at
# the output ( "out"). Unlike P_TO_SET I may be able to do with that tag but it doesn't matter
# too much if it doesn't. Also unlike P_TO_SET we will only be dealing with electric power (
# though it could be extended with I and V being replace by a quantity of effort and a quantity of
# flow). Should be very similar to V_TO_SET
I_TO_SET = "current_to_precondition"
# SFR contains a boolean which tells whether or not this components requires the position of
# flaps for the computation of the slipstream effects.
SFR = "slipstream_flaps_required"
# SWL contains a boolean which tells whether or not this components increase in lift coefficient
# contributes to the wing lift. The reason being the computation of the induced drag need the
# square of the increase in lift coefficient hence they must be summed beforehand
SWL = "slipstream_contributes_to_wing_lift"
# DST_W contains a list of position for which the component should be considered as a distributed
# load that acts on the wing
DST_W = "distributed_wing"
# PCT_W contains a list of position for which the component should be considered as a punctual
# load that acts on the wing
PCT_W = "punctual_wing"
# VARIES_MASS contains a boolean which tells whether or not the components makes the mass of the
# aircraft vary during the mission. This will help prevent setting a fake initial fuel
# consumption when the power train is all electric. DST_W_F contains a list of position for which
# the component should be considered as a distributed load that acts on the wing and contains
# fuel. (The reason being that to compute the stresses on the wing we need to test with and
# without fuel)
DST_W_F = "distributed_fuel_wing"
# PCT_W contains a list of position for which the component should be considered as a punctual
# load that acts on the wing
PCT_W_F = "punctual_fuel_wing"
VARIES_MASS = "varies_mass"
# VARIESN_T_MASS contains a boolean which tells whether or not the component is a source
# component which does not make the mass of the aircraft
VARIESN_T_MASS = "unconsumable_source"
# ETA contains an assumed efficiency for the initial guess of the power and current all along the
# power train
ETA = "efficiency"

PROPELLER = {
    ID: "fastga_he.pt_component.propeller",
    CN: "Propeller",
    CN_ID: "propeller_id",
    CT: "propeller",
    ATT: None,
    PT: ["convergence:*", "true_airspeed", "altitude", "density", "settings:*"],
    SPT: ["data:*", "true_airspeed", "cl_wing_clean", "density", "alpha"],
    PTS: [],
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
    SMP: [
        {"delta_Cd": None},
        {"delta_Cl": None},
        {"delta_Cm": None},
    ],
    ICON: "propeller",
    ICON_SIZE: 30,
    RSD: ["thrust_coefficient", "power_coefficient"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [("shaft_power_in", "in")],
    I_TO_SET: [],
    SFR: True,
    SWL: True,
    DST_W: [],
    PCT_W: ["on_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.8,
}
PMSM = {
    ID: "fastga_he.pt_component.pmsm",
    CN: "PMSM",
    CN_ID: "motor_id",
    CT: "PMSM",
    ATT: None,
    PT: ["settings:*"],
    SPT: [],
    PTS: [],
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
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "e_motor",
    ICON_SIZE: 40,
    RSD: ["ac_current_rms_in", "ac_voltage_rms_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["ac_voltage_rms_in", "ac_voltage_peak_in"],
    P_TO_SET: [("active_power", "in")],
    I_TO_SET: [("ac_current_rms_in", "in"), ("ac_current_rms_in_one_phase", "in")],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["on_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.95,
}
INVERTER = {
    ID: "fastga_he.pt_component.inverter",
    CN: "Inverter",
    CN_ID: "inverter_id",
    CT: "inverter",
    ATT: None,
    PT: ["settings:*"],
    SPT: [],
    PTS: [],
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
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "inverter",
    ICON_SIZE: 30,
    RSD: ["modulation_index", "dc_current_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [],
    I_TO_SET: [("dc_current_in", "in")],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.98,
}
DC_BUS = {
    ID: "fastga_he.pt_component.dc_bus",
    CN: "DCBus",
    CN_ID: "dc_bus_id",
    CT: "DC_bus",
    ATT: ["number_of_inputs", "number_of_outputs"],
    PT: [],
    SPT: [],
    PTS: [],
    IN: [(None, "dc_voltage"), ("dc_current_in_", None)],
    OUT: [(None, "dc_voltage"), ("dc_current_out_", None)],
    CTC: "connector",
    MP: [{"dc_voltage": "V"}],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "bus_bar",
    ICON_SIZE: 20,
    RSD: ["dc_voltage"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["dc_voltage"],
    P_TO_SET: [],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 1.0,
}
DC_LINE = {
    ID: "fastga_he.pt_component.dc_line",
    CN: "Harness",
    CN_ID: "harness_id",
    CT: "DC_cable_harness",
    ATT: None,
    PT: ["exterior_temperature", "settings:*", "time_step"],
    SPT: [],
    PTS: [],
    IN: [("dc_voltage_in", None), (None, "dc_current")],
    OUT: [("dc_voltage_out", None), (None, "dc_current")],
    CTC: "connector",
    MP: [{"dc_current": "A"}, {"cable_temperature": "degK"}, {"conduction_losses": "W"}],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "cable",
    ICON_SIZE: 30,
    RSD: ["dc_current"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [],
    I_TO_SET: [("dc_current_one_cable", "out")],  # Could really be "in" or "out" its the same value
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: [],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.98,
}
DC_DC_CONVERTER = {
    ID: "fastga_he.pt_component.dc_dc_converter",
    CN: "DCDCConverter",
    CN_ID: "dc_dc_converter_id",
    CT: "DC_DC_converter",
    ATT: None,
    PT: [],
    SPT: [],
    PTS: [],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [("dc_voltage_out", None), (None, "dc_current_out")],
    CTC: "connector",
    MP: [
        {"efficiency": None},
        {"duty_cycle": None},
        {"dc_current_in": "A"},
        {"dc_current_out": "A"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "dc_converter",
    ICON_SIZE: 30,
    RSD: ["dc_current_in", "dc_current_out", "duty_cycle"],
    SETS_V: True,
    IO_INDEP_V: True,
    V_TO_SET: [],  # It is a bit paradoxical but you cant set a setter's voltage :p
    P_TO_SET: [("converter_relation.power_rel", "in")],
    I_TO_SET: [("dc_current_in", "in"), ("dc_current_out", "out")],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.98,
}
BATTERY_PACK = {
    ID: "fastga_he.pt_component.battery_pack",
    CN: "BatteryPack",
    CN_ID: "battery_pack_id",
    CT: "battery_pack",
    ATT: None,
    PT: ["time_step"],
    SPT: [],
    PTS: [],
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
        {"power_out": "kW"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "battery",
    ICON_SIZE: 40,
    RSD: ["voltage_out", "state_of_charge", "c_rate"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [("power_out", "out")],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: ["inside_the_wing"],
    PCT_W: ["wing_pod"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: True,
    ETA: 0.95,
}
DC_SSPC = {
    ID: "fastga_he.pt_component.dc_sspc",
    CN: "DCSSPC",
    CN_ID: "dc_sspc_id",
    CT: "DC_SSPC",
    ATT: ["closed_by_default"],
    PT: [],
    SPT: [],
    PTS: [],
    IN: [("dc_voltage_in", None), (None, "dc_current_in")],
    OUT: [(None, "dc_voltage_out"), ("dc_current_out", None)],
    CTC: "connector",
    MP: [
        {"dc_current_in": "A"},
        {"efficiency": None},
        {"power_losses": "W"},
        {"dc_voltage_out": "V"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "switch",
    ICON_SIZE: 2,
    RSD: ["dc_voltage_out", "dc_current_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["dc_voltage_out"],
    P_TO_SET: [("power_flow", "in")],
    I_TO_SET: [("dc_current_in", "in")],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.99,
}
DC_SPLITTER = {
    ID: "fastga_he.pt_component.dc_splitter",
    CN: "DCSplitter",
    CN_ID: "dc_splitter_id",
    CT: "DC_splitter",
    ATT: ["splitter_mode"],
    PT: [],
    SPT: [],
    PTS: [],
    IN: [(None, "dc_voltage_in_"), ("dc_current_in_", None)],
    OUT: [(None, "dc_voltage"), ("dc_current_out", None)],
    CTC: "connector",
    MP: [{"dc_voltage": "V"}, {"power_split": "percent"}],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "splitter",
    ICON_SIZE: 20,
    RSD: ["power_split", "dc_voltage"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: ["dc_voltage", "dc_voltage_in_1", "dc_voltage_in_2"],
    P_TO_SET: [],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 1.0,
}
RECTIFIER = {
    ID: "fastga_he.pt_component.rectifier",
    CN: "Rectifier",
    CN_ID: "rectifier_id",
    CT: "rectifier",
    ATT: None,
    PT: [],
    SPT: [],
    PTS: [],
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
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "rectifier",
    ICON_SIZE: 30,
    RSD: ["dc_current_out", "modulation_index", "ac_current_rms_in_one_phase"],
    SETS_V: True,
    IO_INDEP_V: True,
    V_TO_SET: [],
    P_TO_SET: [("converter_relation.power_rel", "in")],
    I_TO_SET: [("ac_current_rms_in_one_phase", "in"), ("dc_current_out", "out")],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.98,
}
GENERATOR = {
    ID: "fastga_he.pt_component.generator",
    CN: "Generator",
    CN_ID: "generator_id",
    CT: "generator",
    ATT: None,
    PT: [],
    SPT: [],
    PTS: [],
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
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "generator",
    ICON_SIZE: 30,
    RSD: ["shaft_power_in", "ac_voltage_rms_out", "efficiency"],
    SETS_V: True,
    IO_INDEP_V: False,
    V_TO_SET: ["ac_voltage_rms_out", "ac_voltage_peak_out"],
    P_TO_SET: [("active_power", "out"), ("shaft_power_in", "in")],
    I_TO_SET: [("ac_current_rms_out", "out")],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.95,
}
ICE = {
    ID: "fastga_he.pt_component.internal_combustion_engine",
    CN: "ICE",
    CN_ID: "ice_id",
    CT: "ICE",
    ATT: None,
    PT: ["time_step", "density", "settings:*"],
    SPT: [],
    PTS: [],
    IN: [(None, "fuel_consumed_t")],
    OUT: [("rpm", None), ("shaft_power_out", None)],
    CTC: ["source", "propulsive_load"],
    MP: [
        {"torque_out": "N*m"},
        {"specific_fuel_consumption": "kg/kW/h"},
        {"mean_effective_pressure": "bar"},
        {"fuel_consumption": "kg/h"},
        {"fuel_consumed_t": "kg"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "ice",
    ICON_SIZE: 40,
    RSD: ["fuel_consumption", "mean_effective_pressure", "torque_out"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["on_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: True,
    VARIESN_T_MASS: False,
    ETA: 0.4,
}
FUEL_TANK = {
    ID: "fastga_he.pt_component.fuel_tank",
    CN: "FuelTank",
    CN_ID: "fuel_tank_id",
    CT: "fuel_tank",
    ATT: None,
    PT: [],
    SPT: [],
    PTS: [],
    IN: None,
    OUT: [("fuel_consumed_t", None)],
    CTC: "tank",
    MP: [
        {"fuel_remaining_t": "kg"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "fuel_tank",
    ICON_SIZE: 30,
    RSD: ["fuel_remaining_t"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: [],
    DST_W_F: ["inside_the_wing"],
    PCT_W_F: ["wing_pod"],
    VARIES_MASS: False,  # Seems weird but the ICE already does the job so we won't double up
    VARIESN_T_MASS: True,
    ETA: 1.0,
}
FUEL_SYSTEM = {
    ID: "fastga_he.pt_component.fuel_system",
    CN: "FuelSystem",
    CN_ID: "fuel_system_id",
    CT: "fuel_system",
    ATT: ["number_of_engines", "number_of_tanks"],
    PT: [],
    SPT: [],
    PTS: [],
    IN: [(None, "fuel_consumed_in_t_")],
    OUT: [("fuel_consumed_out_t_", None)],
    CTC: "connector",
    MP: [
        {"fuel_flowing_t": "kg"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "fuel_system",
    ICON_SIZE: 30,
    RSD: ["fuel_flowing_t"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: [],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,  # Seems weird but the ICE already does the job so we won't double up
    VARIESN_T_MASS: True,
    ETA: 1.0,
}
TURBOSHAFT = {
    ID: "fastga_he.pt_component.turboshaft",
    CN: "Turboshaft",
    CN_ID: "turboshaft_id",
    CT: "turboshaft",
    ATT: None,
    PT: ["time_step", "density", "settings:*", "altitude", "true_airspeed"],
    SPT: ["data:*", "true_airspeed", "density", "altitude"],
    PTS: ["shaft_power_out"],
    IN: [(None, "fuel_consumed_t")],
    OUT: [("rpm", None), ("shaft_power_out", None)],
    CTC: ["source", "propulsive_load"],
    MP: [
        {"specific_fuel_consumption": "kg/kW/h"},
        {"fuel_consumption": "kg/h"},
        {"fuel_consumed_t": "kg"},
        {"design_power_opr_limit": "kW"},
        {"design_power_itt_limit": "kW"},
    ],
    SMP: [
        {"exhaust_velocity": "m/s"},
        {"exhaust_mass_flow": "kg/s"},
        {"exhaust_thrust": "N"},
        {"delta_Cd": None},
    ],
    ICON: "turbine",
    ICON_SIZE: 30,
    RSD: ["fuel_consumption", "torque_out"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["on_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: True,
    VARIESN_T_MASS: False,
    ETA: 0.35,
}
SPEED_REDUCER = {
    ID: "fastga_he.pt_component.speed_reducer",
    CN: "SpeedReducer",
    CN_ID: "speed_reducer_id",
    CT: "speed_reducer",
    ATT: None,
    PT: [],
    SPT: [],
    PTS: [],
    IN: [(None, "rpm_in"), (None, "shaft_power_in")],
    OUT: [("rpm_out", None), ("shaft_power_out", None)],
    CTC: "connector",
    MP: [
        {"torque_in": "N*m"},
        {"torque_out": "N*m"},
        {"rpm_in": "min**-1"},
        {"shaft_power_in": "kW"},
    ],
    SMP: [
        {"delta_Cd": None},
    ],
    ICON: "gearbox",
    ICON_SIZE: 30,
    RSD: ["torque_in"],
    SETS_V: False,
    IO_INDEP_V: False,
    V_TO_SET: [],
    P_TO_SET: [("shaft_power_in", "in")],
    I_TO_SET: [],
    SFR: False,
    SWL: False,
    DST_W: [],
    PCT_W: ["inside_the_wing"],
    DST_W_F: [],
    PCT_W_F: [],
    VARIES_MASS: False,
    VARIESN_T_MASS: False,
    ETA: 0.98,
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
    FUEL_TANK,
    FUEL_SYSTEM,
    TURBOSHAFT,
    SPEED_REDUCER,
]

KNOWN_ID = []

DICTIONARY_CN = {}
DICTIONARY_CN_ID = {}
DICTIONARY_CT = {}
DICTIONARY_ATT = {}
DICTIONARY_PT = {}
DICTIONARY_SPT = {}
DICTIONARY_PTS = {}
DICTIONARY_IN = {}
DICTIONARY_OUT = {}
DICTIONARY_CTC = {}
DICTIONARY_MP = {}
DICTIONARY_SMP = {}
DICTIONARY_ICON = {}
DICTIONARY_ICON_SIZE = {}
DICTIONARY_RSD = {}
DICTIONARY_SETS_V = {}
DICTIONARY_IO_INDEP_V = {}
DICTIONARY_V_TO_SET = {}
DICTIONARY_P_TO_SET = {}
DICTIONARY_I_TO_SET = {}
DICTIONARY_SFR = {}
DICTIONARY_SWL = {}
DICTIONARY_DST_W = {}
DICTIONARY_DST_W_F = {}
DICTIONARY_PCT_W = {}
DICTIONARY_PCT_W_F = {}
DICTIONARY_VARIES_MASS = {}
DICTIONARY_VARIESN_T_MASS = {}
DICTIONARY_ETA = {}

for known_component in KNOWN_COMPONENTS:
    KNOWN_ID.append(known_component[ID])
    DICTIONARY_CN[known_component[ID]] = known_component[CN]
    DICTIONARY_CN_ID[known_component[ID]] = known_component[CN_ID]
    DICTIONARY_CT[known_component[ID]] = known_component[CT]
    DICTIONARY_ATT[known_component[ID]] = known_component[ATT]
    DICTIONARY_PT[known_component[ID]] = known_component[PT]
    DICTIONARY_SPT[known_component[ID]] = known_component[SPT]
    DICTIONARY_PTS[known_component[ID]] = known_component[PTS]
    DICTIONARY_IN[known_component[ID]] = known_component[IN]
    DICTIONARY_OUT[known_component[ID]] = known_component[OUT]
    DICTIONARY_CTC[known_component[ID]] = known_component[CTC]
    DICTIONARY_MP[known_component[ID]] = known_component[MP]
    DICTIONARY_SMP[known_component[ID]] = known_component[SMP]
    DICTIONARY_ICON[known_component[ID]] = known_component[ICON]
    DICTIONARY_ICON_SIZE[known_component[ID]] = known_component[ICON_SIZE]
    DICTIONARY_RSD[known_component[ID]] = known_component[RSD]
    DICTIONARY_SETS_V[known_component[ID]] = known_component[SETS_V]
    DICTIONARY_IO_INDEP_V[known_component[ID]] = known_component[IO_INDEP_V]
    DICTIONARY_V_TO_SET[known_component[ID]] = known_component[V_TO_SET]
    DICTIONARY_P_TO_SET[known_component[ID]] = known_component[P_TO_SET]
    DICTIONARY_I_TO_SET[known_component[ID]] = known_component[I_TO_SET]
    DICTIONARY_SFR[known_component[ID]] = known_component[SFR]
    DICTIONARY_SWL[known_component[ID]] = known_component[SWL]
    DICTIONARY_DST_W[known_component[ID]] = known_component[DST_W]
    DICTIONARY_DST_W_F[known_component[ID]] = known_component[DST_W_F]
    DICTIONARY_PCT_W[known_component[ID]] = known_component[PCT_W]
    DICTIONARY_PCT_W_F[known_component[ID]] = known_component[PCT_W_F]
    DICTIONARY_VARIES_MASS[known_component[ID]] = known_component[VARIES_MASS]
    DICTIONARY_VARIESN_T_MASS[known_component[ID]] = known_component[VARIESN_T_MASS]
    DICTIONARY_ETA[known_component[ID]] = known_component[ETA]

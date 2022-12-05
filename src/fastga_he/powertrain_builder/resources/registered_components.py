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
ATT = "attributes"
IN = "inputs"
OUT = "outputs"

PROPELLER = {
    ID: "fastga_he.pt_component.propeller",
    CN: "Propeller",
    CN_ID: "propeller_id",
    CT: "propeller",
    ATT: None,
}
PMSM = {
    ID: "fastga_he.pt_component.pmsm",
    CN: "PMSM",
    CN_ID: "motor_id",
    CT: "PMSM",
    ATT: None,
}
INVERTER = {
    ID: "fastga_he.pt_component.inverter",
    CN: "Inverter",
    CN_ID: "inverter_id",
    CT: "inverter",
    ATT: None,
}
DC_BUS = {
    ID: "fastga_he.pt_component.dc_bus",
    CN: "DC_BUS",
    CN_ID: "dc_bus_id",
    CT: "DC_bus",
    ATT: ["number_of_inputs", "number_of_outputs"],
}
DC_LINE = {
    ID: "fastga_he.pt_component.dc_line",
    CN: "Harness",
    CN_ID: "harness_id",
    CT: "DC_cable_harness",
    ATT: None,
}
DC_DC_CONVERTER = {
    ID: "fastga_he.pt_component.dc_dc_converter",
    CN: "DCDCConverter",
    CN_ID: "dc_dc_converter_id",
    CT: "DC_DC_converter",
    ATT: None,
}
BATTERY_PACK = {
    ID: "fastga_he.pt_component.battery_pack",
    CN: "BatteryPack",
    CN_ID: "battery_pack_id",
    CT: "battery_pack",
    ATT: None,
}

KNOWN_COMPONENTS = [PROPELLER, PMSM, INVERTER, DC_BUS, DC_LINE, DC_DC_CONVERTER, BATTERY_PACK]

KNOWN_ID = []

DICTIONARY_CN = {}
DICTIONARY_CN_ID = {}
DICTIONARY_CT = {}
DICTIONARY_ATT = {}

for known_component in KNOWN_COMPONENTS:
    KNOWN_ID.append(known_component[ID])
    DICTIONARY_CN[known_component[ID]] = known_component[CN]
    DICTIONARY_CN_ID[known_component[ID]] = known_component[CN_ID]
    DICTIONARY_CT[known_component[ID]] = known_component[CT]
    DICTIONARY_ATT[known_component[ID]] = known_component[ATT]

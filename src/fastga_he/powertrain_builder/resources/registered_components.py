# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

ID = "id"
ATT = "attributes"

PROPELLER = {ID: "fastga_he.pt_component.propeller", ATT: ["rpm_mission"]}
PMSM = {ID: "fastga_he.pt_component.pmsm", ATT: None}
INVERTER = {
    ID: "fastga_he.pt_component.inverter",
    ATT: ["switching_frequency", "heat_sink_temperature"],
}
DC_BUS = {ID: "fastga_he.pt_component.dc_bus", ATT: ["number_of_inputs", "number_of_outputs"]}
DC_LINE = {ID: "fastga_he.pt_component.dc_line", ATT: None}
DC_DC_CONVERTER = {
    ID: "fastga_he.pt_component.dc_dc_converter",
    ATT: ["switching_frequency", "voltage_out_target"],
}
BATTERY_PACK = {ID: "fastga_he.pt_component.battery_pack", ATT: ["cell_temperature"]}

KNOWN_COMPONENTS = [PROPELLER, PMSM, INVERTER, DC_BUS, DC_LINE, DC_DC_CONVERTER, BATTERY_PACK]

KNOWN_ID = []

for known_component in KNOWN_COMPONENTS:
    KNOWN_ID.append(known_component[ID])

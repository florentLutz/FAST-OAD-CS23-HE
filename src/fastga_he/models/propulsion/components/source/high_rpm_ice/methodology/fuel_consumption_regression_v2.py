# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import plotly.graph_objects as go

if __name__ == "__main__":
    # We'll focus on max continuous performances, 5500 seems to be a common upper limit for
    # continuous operation
    mcp_914 = 73.5  # kW at 5500 rpm
    mcp_912a = 58.0  # kW at 5500 rpm
    mcp_912s = 69.0  # kW at 5500 rpm

    to_912a = 59.6  # kW at 5800 rpm
    to_912s = 73.5  # kW at 5800 rpm

    percent_75_912a = 43.5  # kW at 5000 rpm
    percent_75_912s = 51.0  # kW at 5000 rpm

    # Volumetric fuel consumption in l/h
    vfc_mcp_914 = 26.0  # Can't find the exact value :/
    vfc_mcp_912a = 22.6
    vfc_mcp_912s = 25.0

    vfc_to_912a = 24.0
    vfc_to_912s = 27.0

    vfc_75_912a = 16.2
    vfc_75_912s = 18.5

    avgas_density = 0.72  # in kg/l

    sfc_mcp_912a = vfc_mcp_912a * avgas_density / mcp_912a * 1000.0  # In g/kWh
    sfc_mcp_912s = vfc_mcp_912s * avgas_density / mcp_912s * 1000.0  # In g/kWh

    sfc_to_912a = vfc_to_912a * avgas_density / to_912a * 1000.0  # In g/kWh
    sfc_to_912s = vfc_to_912s * avgas_density / to_912s * 1000.0  # In g/kWh

    sfc_75_912a = vfc_75_912a * avgas_density / percent_75_912a * 1000.0  # In g/kWh
    sfc_75_912s = vfc_75_912s * avgas_density / percent_75_912s * 1000.0  # In g/kWh

    print("Rotax 912-A 75%", sfc_75_912a)
    print("Rotax 912-A MCP", sfc_mcp_912a)
    print("Rotax 912-A TO", sfc_to_912a)

    print("Rotax 912-S 75%", sfc_75_912s)
    print("Rotax 912-S MCP", sfc_mcp_912s)  # The operator manual announce 285 g/kWh ...
    print("Rotax 912-S TO", sfc_to_912s)

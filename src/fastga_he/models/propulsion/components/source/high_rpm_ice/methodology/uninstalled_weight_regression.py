# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

if __name__ == "__main__":
    mcp_914 = 73.5  # kW at 5500 rpm
    mcp_912a = 58.0  # kW at 5500 rpm
    mcp_912s = 69.0  # kW at 5500 rpm

    # Only engine is considered here as the rest will be accounted for in the installation penalty
    weight_914 = 64.4  # Turbocharged.
    weight_912a = 55.4
    weight_912s = 56.6

    # We'll assume alinear regression like what is done for Lycoming type engine

    print("a:", (55.4 - 56.6) / (58.0 - 69.0))
    print("b:", 55.4)

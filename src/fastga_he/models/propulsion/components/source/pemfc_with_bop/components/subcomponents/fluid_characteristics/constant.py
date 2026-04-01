# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

fluid_name_dict = {
    "air": "air",
    "water": "water",
    "hydrogen": "hydrogen",
    "ammonia": "ammonia",
    "ethylene glycol": "INCOMP::MEG-50%",
    "propylene glycol": "INCOMP::MPG-50%",
    "potassium formate": "INCOMP::MKF-40%",
    "R134a": "R134a",
}

fluid_density_dict = {
    "air": 1.177,
    "water": 996.557,
    "hydrogen": 0.0818,
    "ammonia": 0.699,
    "ethylene glycol": 1061.179,
    "propylene glycol": 1034.547,
    "potassium formate": 1258.802,
    "R134a": 4.23,
}  # [kg/m**2]

fluid_dynamic_viscosity_dict = {
    "air": 1.854e-5,
    "water": 0.000854,
    "hydrogen": 8.938e-4,
    "ammonia": 1.016e-5,
    "ethylene glycol": 0.00299,
    "propylene glycol": 0.00474,
    "potassium formate": 0.00176,
    "R134a": 1.189e-5,
}  # [Pa*s]

fluid_enthalpy_dict = {
    "air": 426297.774,
    "water": 112654.9,
    "hydrogen": 3958280.602,
    "ammonia": 1696298.936,
    "ethylene glycol": 22808.858,
    "propylene glycol": 24271.052,
    "potassium formate": 19756.474,
    "R134a": 426102.82,
}  # [J/kg]

fluid_prandtl_number_dict = {
    "air": 0.707,
    "water": 5.856,
    "hydrogen": 0.685,
    "ammonia": 0.876,
    "ethylene glycol": 25.42,
    "propylene glycol": 46.438,
    "potassium formate": 9.709,
    "R134a": 0.75,
}

fluid_specific_heat_capacity_dict = {
    "air": 1006.374,
    "water": 4180.636,
    "hydrogen": 14312.822,
    "ammonia": 2163.385,
    "ethylene glycol": 3347.568,
    "propylene glycol": 3556.523,
    "potassium formate": 2890.593,
    "R134a": 854.03,
}  # [J/kg/K]

fluid_thermal_conductivity_dict = {
    "air": 0.0264,
    "water": 0.61,
    "hydrogen": 0.187,
    "ammonia": 0.0251,
    "ethylene glycol": 0.393,
    "propylene glycol": 0.363,
    "potassium formate": 0.524,
    "R134a": 0.0135,
}  # [W/m/K]

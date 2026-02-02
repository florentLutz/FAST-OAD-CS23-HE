# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO
from stdatm import AtmosphereWithPartials

SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA = "submodel.propulsion.constraints.pemfc.effective_area"
SUBMODEL_CONSTRAINTS_PEMFC_POWER = "submodel.propulsion.constraints.pemfc.power"

POSSIBLE_POSITION = ["in_the_front", "wing_pod", "underbelly", "in_the_back"]

DEFAULT_PRESSURE = AtmosphereWithPartials(0).pressure  # [Pa]
FARADAY_CONSTANT = 96485.3321  # [C/mol]
GAS_CONSTANT = 8.314  # [J/(mol*K)]
H2_MOL_PER_KG = 500.0  # [mol/kg]
NUMBER_OF_ELETRONS_FROM_H2 = 2.0
MAX_DEFAULT_POWER = 100.0  # [kW]
MAX_DEFAULT_CURRENT = 1000.0  # [A]
HHV_HYDROGEN_EQUIVALENT_VOLTAGE = 1.481  # [V]
REVERSIBLE_ELECTRIC_POTENTIAL = 1.229  # [V]
MAX_CURRENT_DENSITY_EMPIRICAL = 0.7  # [A/cm^2]
MAX_CURRENT_DENSITY_ANALYTICAL = 1.95  # [A/cm^2]
FUEL_UTILIZATION_COEFFICIENT = 0.95
DEFAULT_LAYER_VOLTAGE = 0.7  # [V]
DEFAULT_PRESSURE_ATM = 1.0  # [atm]
DEFAULT_LAYER_TEMPERATURE = 288.15  # [K]
DEFAULT_PEMFC_EFFICIENCY = 0.53
DEFAULT_HYDROGEN_CONSUMPTION = 4.2  # [kg/h]
DEFAULT_FC_SPECIFIC_POWER = 0.345  # [kW/kg]

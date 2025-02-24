# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
from stdatm import AtmosphereWithPartials

SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA = "submodel.propulsion.constraints.pemfc.effective_area"
SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE = "submodel.propulsion.performances.pemfc.layer_voltage"

POSSIBLE_POSITION = ["in_the_front", "wing_pod", "underbelly", "in_the_back"]

DEFAULT_PRESSURE = AtmosphereWithPartials(0).pressure  # [Pa]
FARADAY_CONSTANT = 96485.3321  # [C/mol]
GAS_CONSTANT = 8.314  # [J/(mol*K)]
H2_MOL_PER_KG = 500.0  # [mol/kg]
DEFAULT_TEMPERATURE = AtmosphereWithPartials(0).temperature  # [K]
NUMBER_OF_ELETRONS_FROM_H2 = 2.0
MAX_DEFAULT_STACK_POWER = 1000.0  # [kW]
MAX_DEFAULT_STACK_CURRENT = 1000.0  # [A]
HHV_HYDROGEN_EQUIVALENT_VOLTAGE = 1.481  # [V]
REVERSIBLE_ELECTRIC_POTENTIAL = 1.229  # [V]
MAX_CURRENT_DENSITY_EMPIRICAL = 0.7  # [A/cm^2]
MAX_CURRENT_DENSITY_ANALYTICAL = 2.0  # [A/cm^2]

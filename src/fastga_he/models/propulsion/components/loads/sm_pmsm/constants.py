# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
from stdatm import AtmosphereWithPartials
import numpy as np

SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE = "submodel.propulsion.constraints.sm_pmsm.torque"
SUBMODEL_CONSTRAINTS_SM_PMSM_RPM = "submodel.propulsion.constraints.sm_pmsm.rpm"
SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE = "submodel.propulsion.constraints.sm_pmsm.voltage"

POSSIBLE_POSITION = ["on_the_wing", "in_the_nose"]
IRON_LOSSES_COEFF = [
    [5.30850444e2, -1.66022877e3, 1.67666819e3, -5.40045900e2],
    [-4.04065802e1, 1.26706523e2, -1.27987721e2, 4.10664456e1],
    [8.43378999e-1, -2.63865343e0, 2.65237021e0, -8.40175850e-1],
    [-4.35714286e-3, 1.35660947e-2, -1.35585345e-2, 4.25562126e-3],
]

DEFAULT_DENSITY = AtmosphereWithPartials(0).density  # [kg/m^3]
DEFAULT_DYNAMIC_VISCOSITY = AtmosphereWithPartials(0).dynamic_viscosity  # [kg/m/s]
VACUUM_MAGNETIC_PERMEABILITY = 4.0 * np.pi * 1e-7  # [H/m]

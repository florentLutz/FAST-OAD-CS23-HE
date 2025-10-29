# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
from stdatm import AtmosphereWithPartials
import numpy as np

SUBMODEL_CONSTRAINTS_SM_PMSM_TORQUE = "submodel.propulsion.constraints.sm_pmsm.torque"
SUBMODEL_CONSTRAINTS_SM_PMSM_RPM = "submodel.propulsion.constraints.sm_pmsm.rpm"
SUBMODEL_CONSTRAINTS_SM_PMSM_VOLTAGE = "submodel.propulsion.constraints.sm_pmsm.voltage"

POSSIBLE_POSITION = ["on_the_wing", "in_the_nose"]

DEFAULT_DYNAMIC_VISCOSITY = AtmosphereWithPartials(0).dynamic_viscosity  # [kg/m/s]
VACUUM_MAGNETIC_PERMEABILITY = 4.0 * np.pi * 1e-7  # [H/m]

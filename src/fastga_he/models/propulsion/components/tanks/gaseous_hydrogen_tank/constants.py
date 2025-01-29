# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import numpy as np

SUBMODEL_CONSTRAINTS_GASEOUS_HYDROGEN_TANK_CAPACITY = (
    "submodel.propulsion.constraints.gaseous_hydrogen_tank.capacity"
)

POSSIBLE_POSITION = ["in_the_front", "in_the_fuselage", "wing_pod", "in_the_back", "underbelly"]

MULTI_TANK_FACTOR = {
    1: 1,
    2: 2,
    3: 1 + (2 / 3) * np.sqrt(3),
    4: 1 + np.sqrt(2),
    5: 1 + np.sqrt(2 * (1 + 1 / np.sqrt(5))),
    6: 3,
    7: 3,
    8: 1 + 1 / np.sin(np.pi / 7),
    9: 1 + np.sqrt(2 * (2 + np.sqrt(2))),
}

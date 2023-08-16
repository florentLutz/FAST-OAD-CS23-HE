#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .constants import (
    HE_SUBMODEL_AEROSTRUCTURAL_LOADS,
    HE_SUBMODEL_STRUCTURAL_LOADS,
    HE_SUBMODEL_AERODYNAMIC_LOADS,
)


@oad.RegisterOpenMDAOSystem("fastga_he.loads.wing", domain=ModelDomain.OTHER)
class WingLoadsHE(om.Group):
    def setup(self):
        self.add_subsystem(
            "aerostructural_loads",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_AEROSTRUCTURAL_LOADS),
            promotes=["*"],
        )
        self.add_subsystem(
            "structural_loads",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_STRUCTURAL_LOADS),
            promotes=["*"],
        )
        self.add_subsystem(
            "aerodynamic_loads",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_AERODYNAMIC_LOADS),
            promotes=["*"],
        )

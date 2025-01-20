# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .sizing_tank_unusable_hydrogen import SizingCryogenicHydrogenTankUnusableHydrogen
from .sizing_tank_total_hydrogen_mission import SizingCryogenicHydrogenTankTotalHydrogenMission
from .sizing_tank_wall_thickness import SizingCryogenicHydrogenTankWallThickness
from .sizing_tank_wall_diameter import SizingCryogenicHydrogenTankWallDiameter
from .sizing_tank_cg_x import SizingCryogenicHydrogenTankCGX
from .sizing_tank_cg_y import SizingCryogenicHydrogenTankCGY
from .sizing_tank_length import SizingCryogenicHydrogenTankLength
from .sizing_tank_inner_volume import SizingCryogenicHydrogenTankInnerVolume
from .sizing_tank_inner_diameter import SizingCryogenicHydrogenTankInnerDiameter
from .sizing_tank_weight import SizingCryogenicHydrogenTankWeight
from .sizing_gravimetric_index import SizingCryogenicHydrogenTankGravimetricIndex
from .sizing_tank_drag import SizingCryogenicHydrogenTankDrag
from .sizing_tank_aspect_ratio import SizingCryogenicHydrogenTankAspectRatio
from .sizing_tank_stress_coefficient import SizingCryogenicHydrogenTankStressCoefficinet
from .sizing_tank_outer_diameter import SizingCryogenicHydrogenTankOuterDiameter
from .sizing_tank_diameter_update import SizingCryogenicHydrogenTankDiameterUpdate
from .sizing_tank_overall_length import SizingCryogenicHydrogenTankOverallLength
from .sizing_tank_insulation_layer_thermal_resistance import (
    SizingCryogenicHydrogenTankInsulationThermalResistance,
)
from .sizing_tank_wall_thermal_resistance import SizingCryogenicHydrogenTankWallThermalResistance
from .sizing_tank_thermal_resistance import SizingCryogenicHydrogenTankThermalResistance

from .sizing_tank_overall_length_fuselage_check import (
    SizingCryogenicHydrogenTankOverallLengthFuselageCheck,
)

from .cstr_cryogenic_hydrogen_tank import ConstraintsCryogenicHydrogenTank

from ..constants import POSSIBLE_POSITION


class SizingCryogenicHydrogenTank(om.Group):
    """
    Class that regroups all the subcomponents for the sizing of the hydrogen gas tank.
    """

    def initialize(self):
        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="position",
            default="in_the_fuselage",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the hydrogen gas tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]

        self.add_subsystem(
            name="tank_outer_diameter",
            subsys=SizingCryogenicHydrogenTankOuterDiameter(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_diameter_update",
            subsys=SizingCryogenicHydrogenTankDiameterUpdate(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="unusable_hydrogen_gas",
            subsys=SizingCryogenicHydrogenTankUnusableHydrogen(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="total_hydrogen_gas",
            subsys=SizingCryogenicHydrogenTankTotalHydrogenMission(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_constraints",
            subsys=ConstraintsCryogenicHydrogenTank(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_wall_diameter",
            subsys=SizingCryogenicHydrogenTankWallDiameter(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_aspect_ratio",
            subsys=SizingCryogenicHydrogenTankAspectRatio(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_stress_coefficient",
            subsys=SizingCryogenicHydrogenTankStressCoefficinet(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_diameter",
            subsys=SizingCryogenicHydrogenTankInnerDiameter(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_wall_thickness",
            subsys=SizingCryogenicHydrogenTankWallThickness(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_inner_volume",
            subsys=SizingCryogenicHydrogenTankInnerVolume(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_length",
            subsys=SizingCryogenicHydrogenTankLength(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_overall_length",
            subsys=SizingCryogenicHydrogenTankOverallLength(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_x",
            subsys=SizingCryogenicHydrogenTankCGX(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_cg_y",
            subsys=SizingCryogenicHydrogenTankCGY(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id, position=position
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_overall_length_length_fuselage_check",
            subsys=SizingCryogenicHydrogenTankOverallLengthFuselageCheck(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
                position=position,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="insulation_thermal_resistance",
            subsys=SizingCryogenicHydrogenTankInsulationThermalResistance(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="wall_thermal_resistance",
            subsys=SizingCryogenicHydrogenTankWallThermalResistance(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_thermal_resistance",
            subsys=SizingCryogenicHydrogenTankThermalResistance(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_weight_lh2",
            subsys=SizingCryogenicHydrogenTankWeight(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            name="tank_gravimetric_index",
            subsys=SizingCryogenicHydrogenTankGravimetricIndex(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id
            ),
            promotes=["*"],
        )

        for low_speed_aero in [True, False]:
            system_name = "tank_drag_ls" if low_speed_aero else "tank_drag_cruise"
            self.add_subsystem(
                name=system_name,
                subsys=SizingCryogenicHydrogenTankDrag(
                    cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
                    position=position,
                    low_speed_aero=low_speed_aero,
                ),
                promotes=["*"],
            )

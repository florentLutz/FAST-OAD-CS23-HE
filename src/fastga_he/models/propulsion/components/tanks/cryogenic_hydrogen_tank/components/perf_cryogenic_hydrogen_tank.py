# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..components.perf_fuel_mission_consumed import PerformancesLiquidHydrogenConsumedMission
from ..components.perf_fuel_remaining import PerformancesLiquidHydrogenRemainingMission
from ..components.perf_fuel_boil_off import PerformancesHydrogenBoilOffMission
from ..components.perf_exterior_temperature import PerformancesExteriorTemperature
from ..components.perf_nusselt_number import PerformancesCryogenicHydrogenTankNusseltNumber
from ..components.perf_rayleigh_number import PerformancesCryogenicHydrogenTankRayleighNumber
from ..components.perf_tank_skin_temperature import PerformancesLiquidHydrogenTankSkinTemperature
from ..components.perf_air_kinematic_viscosity import PerformancesAirKinematicViscosity
from ..components.perf_air_conductivity import PerformancesAirThermalConductivity
from ..components.perf_tank_heat_radiation import PerformancesCryogenicHydrogenTankRadiation
from ..components.perf_tank_heat_convection import PerformancesCryogenicHydrogenTankConvection
from ..components.perf_tank_heat_conduction import PerformancesCryogenicHydrogenTankConduction
from ..components.perf_tank_temperature import PerformancesLiquidHydrogenTankTemperature
from ..components.perf_total_boil_off_hydrogen import PerformancesHydrogenBoilOffTotal

from ..constants import POSSIBLE_POSITION


class PerformancesCryogenicHydrogenTank(om.Group):
    """
    Regrouping all the components for the performances of the tank. Note that to limit the work
    to be done for the implementation of fuel tanks, fuel tanks don't output the fuel consumed
    used to iterate on the mass during the mission, but it uses it. Just like for the CG where we
    will output the "varying" part of the CG straight from the mission; we could do the same for
    mass (which may or may not improve the computation time).
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
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

        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        position = self.options["position"]

        self.add_subsystem(
            "liquid_hydrogen_consumed_mission",
            PerformancesLiquidHydrogenConsumedMission(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "exterior_temperature",
            PerformancesExteriorTemperature(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "hydrogen_air_kinematic_viscosity",
            PerformancesAirKinematicViscosity(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "hydrogen_air_conductivity",
            PerformancesAirThermalConductivity(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "tank_temperature",
            PerformancesLiquidHydrogenTankTemperature(
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "tank_skin_temperature",
            PerformancesLiquidHydrogenTankSkinTemperature(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "rayleigh_number",
            PerformancesCryogenicHydrogenTankRayleighNumber(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "nusselt_number",
            PerformancesCryogenicHydrogenTankNusseltNumber(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
                position=position,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "heat_radiation",
            PerformancesCryogenicHydrogenTankRadiation(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
                position=position,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "heat_convection",
            PerformancesCryogenicHydrogenTankConvection(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "heat_conduction",
            PerformancesCryogenicHydrogenTankConduction(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "hydrogen_boil_off_mission",
            PerformancesHydrogenBoilOffMission(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "liquid_hydrogen_remaining_mission",
            PerformancesLiquidHydrogenRemainingMission(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "liquid_hydrogen_boil_off_overall",
            PerformancesHydrogenBoilOffTotal(
                number_of_points=number_of_points,
                cryogenic_hydrogen_tank_id=cryogenic_hydrogen_tank_id,
            ),
            promotes=["*"],
        )

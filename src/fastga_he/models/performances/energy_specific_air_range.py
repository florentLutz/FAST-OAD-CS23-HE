#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from fastoad.module_management.constants import ModelDomain

from fastga_he.models.environmental_impacts.simple_energy_impact import (
    ENERGY_CONTENT_JET_FUEL,
    KWH_TO_MJ,
)

ENERGY_CONTENT_AVGAS = 43.7  # In M/kg


@oad.RegisterOpenMDAOSystem(
    "fastga_he.performances.energy_specific_air_range", domain=ModelDomain.PERFORMANCE
)
class EnergySpecificAirRange(om.ExplicitComponent):
    """
    Component that computes the Energy Specific Air Range (ESAR) which is the distance that can be
    travelled with one unit of energy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.energy_content_fuel = None

    def initialize(self):
        mission_possible_option = ["design", "operational", "both"]
        self.options.declare(
            name="mission",
            default="design",
            values=["design", "operational", "both"],
            desc="Option to give the type of mission whose impact we must study, possible option include"
            + ", ".join(mission_possible_option),
            allow_none=False,
        )
        fuel_type_option = ["jet_fuel", "avgas"]
        self.options.declare(
            name="fuel_type",
            default="jet_fuel",
            values=fuel_type_option,
            desc="Option to give the type of fuel used, possible option include"
            + ", ".join(fuel_type_option),
        )

    def setup(self):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            self.add_input("data:mission:sizing:fuel", units="kg", val=np.nan)
            self.add_input("data:mission:sizing:energy", units="kW*h", val=np.nan)

            self.add_input("data:TLAR:range", np.nan, units="NM")

            self.add_output("data:mission:sizing:energy_specific_air_range", units="NM/kW/h")

        if mission_option == "operational" or mission_option == "both":
            self.add_input("data:mission:operational:fuel", units="kg", val=np.nan)
            self.add_input("data:mission:operational:energy", units="kW*h", val=np.nan)

            self.add_input("data:mission:operational:range", np.nan, units="NM")

            self.add_output("data:mission:operational:energy_specific_air_range", units="NM/kW/h")

        if self.options["fuel_type"] == "avgas":
            self.energy_content_fuel = ENERGY_CONTENT_AVGAS
        else:
            self.energy_content_fuel = ENERGY_CONTENT_JET_FUEL

    def setup_partials(self):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            self.declare_partials(
                of="data:mission:sizing:energy_specific_air_range",
                wrt=["data:mission:sizing:fuel", "data:mission:sizing:energy", "data:TLAR:range"],
                method="exact",
            )

        if mission_option == "operational" or mission_option == "both":
            self.declare_partials(
                of="data:mission:operational:energy_specific_air_range",
                wrt=[
                    "data:mission:operational:fuel",
                    "data:mission:operational:energy",
                    "data:mission:operational:range",
                ],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            fuel_consumed = inputs["data:mission:sizing:fuel"]
            electricity_consumed = inputs["data:mission:sizing:energy"]

            design_range = inputs["data:TLAR:range"]

            total_energy = (
                fuel_consumed * self.energy_content_fuel / KWH_TO_MJ + electricity_consumed
            )
            outputs["data:mission:sizing:energy_specific_air_range"] = design_range / total_energy

        if mission_option == "operational" or mission_option == "both":
            fuel_consumed = inputs["data:mission:operational:fuel"]
            electricity_consumed = inputs["data:mission:operational:energy"]

            op_range = inputs["data:mission:operational:range"]

            total_energy = (
                fuel_consumed * self.energy_content_fuel / KWH_TO_MJ + electricity_consumed
            )
            outputs["data:mission:operational:energy_specific_air_range"] = op_range / total_energy

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            fuel_consumed = inputs["data:mission:sizing:fuel"]
            electricity_consumed = inputs["data:mission:sizing:energy"]

            design_range = inputs["data:TLAR:range"]

            total_energy = (
                fuel_consumed * self.energy_content_fuel / KWH_TO_MJ + electricity_consumed
            )

            partials[
                "data:mission:sizing:energy_specific_air_range", "data:mission:sizing:fuel"
            ] = -design_range / total_energy**2.0 * self.energy_content_fuel / KWH_TO_MJ
            partials[
                "data:mission:sizing:energy_specific_air_range", "data:mission:sizing:energy"
            ] = -design_range / total_energy**2.0
            partials["data:mission:sizing:energy_specific_air_range", "data:TLAR:range"] = (
                1.0 / total_energy
            )

        if mission_option == "operational" or mission_option == "both":
            fuel_consumed = inputs["data:mission:operational:fuel"]
            electricity_consumed = inputs["data:mission:operational:energy"]

            op_range = inputs["data:mission:operational:range"]

            total_energy = (
                fuel_consumed * self.energy_content_fuel / KWH_TO_MJ + electricity_consumed
            )
            partials[
                "data:mission:operational:energy_specific_air_range",
                "data:mission:operational:fuel",
            ] = -op_range / total_energy**2.0 * self.energy_content_fuel / KWH_TO_MJ
            partials[
                "data:mission:operational:energy_specific_air_range",
                "data:mission:operational:energy",
            ] = -op_range / total_energy**2.0
            partials[
                "data:mission:operational:energy_specific_air_range",
                "data:mission:operational:range",
            ] = 1.0 / total_energy

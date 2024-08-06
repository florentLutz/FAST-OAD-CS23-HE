# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

CARBON_INTENSITY_JET_FUEL = 88.7  # in gCO2eq/MJ
CARBON_INTENSITY_BIOFUEL_FT = 7.7  # in gCO2eq/MJ
CARBON_INTENSITY_BIOFUEL_HEFA = 21.6  # in gCO2eq/MJ

CARBON_INTENSITY_ELECTRICITY_EUROPE = 72.7  # in gCO2eq/MJ
CARBON_INTENSITY_ELECTRICITY_FRANCE = 29.6  # in gCO2eq/MJ

ENERGY_CONTENT_JET_FUEL = 43.0  # in MJ/kg
ENERGY_CONTENT_BIOFUEL = 43.0  # in MJ/kg, same assumption is taken here as the turboshaft is
# always assumed to run on jet fuel (at least that what its combustion energy says)

KWH_TO_MJ = 3.6


@oad.RegisterOpenMDAOSystem("fastga_he.environmental.energy_simple", domain=ModelDomain.OTHER)
class SimpleEnergyImpacts(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.energy_content_fuel = None
        self.carbon_intensity_electricity = None
        self.carbon_intensity_fuel = None

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
        fuel_type_option = ["jet_fuel", "biofuel_ft_pathway", "biofuel_hefa_pathway"]
        self.options.declare(
            name="fuel_type",
            default="jet_fuel",
            values=fuel_type_option,
            desc="Option to give the type of fuel used, possible option include"
            + ", ".join(fuel_type_option),
        )
        electric_mix_option = ["europe", "france"]
        self.options.declare(
            name="electricity_mix",
            default="europe",
            values=electric_mix_option,
            desc="Option to give the electric mix used, possible option include"
            + ", ".join(electric_mix_option),
        )

    def setup(self):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            self.add_input("data:mission:sizing:fuel", units="kg", val=np.nan)
            self.add_input("data:mission:sizing:energy", units="kW*h", val=np.nan)

            self.add_input("data:TLAR:range", np.nan, units="km")
            self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")

            self.add_output(
                "data:environmental_impact:sizing:fuel_emissions",
                units="g",
                val=0.0,
                desc="Emissions related to fuel consumption and production during the sizing "
                "mission, in gCO2,eq",
            )
            self.add_output(
                "data:environmental_impact:sizing:energy_emissions",
                units="g",
                val=0.0,
                desc="Emissions related to energy production during the sizing mission, in gCO2,eq",
            )
            self.add_output(
                "data:environmental_impact:sizing:emissions",
                units="g",
                val=0.0,
                desc="Total emissions during the sizing mission, in gCO2,eq",
            )
            self.add_output(
                "data:environmental_impact:sizing:emission_factor",
                val=0.0,
                desc="Total emissions during the sizing mission per kg of payload per km [kgCO2/kg/km]",
            )

            self.declare_partials(
                of="data:environmental_impact:sizing:fuel_emissions",
                wrt="data:mission:sizing:fuel",
                method="exact",
            )
            self.declare_partials(
                of="data:environmental_impact:sizing:energy_emissions",
                wrt="data:mission:sizing:energy",
                method="exact",
            )
            self.declare_partials(
                of="data:environmental_impact:sizing:emissions",
                wrt=["data:mission:sizing:fuel", "data:mission:sizing:energy"],
                method="exact",
            )
            self.declare_partials(
                of="data:environmental_impact:sizing:emission_factor",
                wrt=[
                    "data:mission:sizing:fuel",
                    "data:mission:sizing:energy",
                    "data:TLAR:range",
                    "data:weight:aircraft:payload",
                ],
            )

        if mission_option == "operational" or mission_option == "both":
            self.add_input("data:mission:operational:fuel", units="kg", val=np.nan)
            self.add_input("data:mission:operational:energy", units="kW*h", val=np.nan)

            self.add_input("data:mission:operational:range", np.nan, units="km")
            self.add_input("data:mission:operational:payload:mass", val=np.nan, units="kg")

            self.add_output(
                "data:environmental_impact:operational:fuel_emissions",
                units="g",
                val=0.0,
                desc="Emissions related to fuel consumption and production during the operational "
                "mission, in gCO2,eq",
            )
            self.add_output(
                "data:environmental_impact:operational:energy_emissions",
                units="g",
                val=0.0,
                desc="Emissions related to energy production during the operational mission, "
                "in gCO2,eq",
            )
            self.add_output(
                "data:environmental_impact:operational:emissions",
                units="g",
                val=0.0,
                desc="Total emissions during the operational mission, in gCO2,eq",
            )
            self.add_output(
                "data:environmental_impact:operational:emission_factor",
                val=0.0,
                desc="Total emissions during the operational mission per kg of payload per km [kgCO2/kg/km]",
            )

            self.declare_partials(
                of="data:environmental_impact:operational:fuel_emissions",
                wrt="data:mission:operational:fuel",
                method="exact",
            )
            self.declare_partials(
                of="data:environmental_impact:operational:energy_emissions",
                wrt="data:mission:operational:energy",
                method="exact",
            )
            self.declare_partials(
                of="data:environmental_impact:operational:emissions",
                wrt=["data:mission:operational:fuel", "data:mission:operational:energy"],
                method="exact",
            )
            self.declare_partials(
                of="data:environmental_impact:operational:emission_factor",
                wrt=[
                    "data:mission:operational:fuel",
                    "data:mission:operational:energy",
                    "data:mission:operational:range",
                    "data:mission:operational:payload:mass",
                ],
            )

        if self.options["fuel_type"] == "biofuel_ft_pathway":
            self.energy_content_fuel = ENERGY_CONTENT_BIOFUEL
            self.carbon_intensity_fuel = CARBON_INTENSITY_BIOFUEL_FT

        elif self.options["fuel_type"] == "biofuel_hefa_pathway":
            self.energy_content_fuel = ENERGY_CONTENT_BIOFUEL
            self.carbon_intensity_fuel = CARBON_INTENSITY_BIOFUEL_HEFA

        else:
            self.energy_content_fuel = ENERGY_CONTENT_JET_FUEL
            self.carbon_intensity_fuel = CARBON_INTENSITY_JET_FUEL

        if self.options["electricity_mix"] == "france":
            self.carbon_intensity_electricity = CARBON_INTENSITY_ELECTRICITY_FRANCE

        else:
            self.carbon_intensity_electricity = CARBON_INTENSITY_ELECTRICITY_EUROPE

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            fuel_consumed = inputs["data:mission:sizing:fuel"]
            electricity_consumed = inputs["data:mission:sizing:energy"]

            design_range = inputs["data:TLAR:range"]
            payload = inputs["data:weight:aircraft:payload"]

            emissions_fuel = fuel_consumed * self.energy_content_fuel * self.carbon_intensity_fuel
            emissions_electricity = (
                electricity_consumed * self.carbon_intensity_electricity * KWH_TO_MJ
            )

            outputs["data:environmental_impact:sizing:fuel_emissions"] = emissions_fuel
            outputs["data:environmental_impact:sizing:energy_emissions"] = emissions_electricity
            outputs["data:environmental_impact:sizing:emissions"] = (
                emissions_electricity + emissions_fuel
            )
            outputs["data:environmental_impact:sizing:emission_factor"] = (
                (emissions_electricity + emissions_fuel) / design_range / payload
            )

        if mission_option == "operational" or mission_option == "both":
            fuel_consumed = inputs["data:mission:operational:fuel"]
            electricity_consumed = inputs["data:mission:operational:energy"]

            op_range = inputs["data:mission:operational:range"]
            payload = inputs["data:mission:operational:payload:mass"]

            emissions_fuel = fuel_consumed * self.energy_content_fuel * self.carbon_intensity_fuel
            emissions_electricity = (
                electricity_consumed * self.carbon_intensity_electricity * KWH_TO_MJ
            )

            outputs["data:environmental_impact:operational:fuel_emissions"] = emissions_fuel
            outputs["data:environmental_impact:operational:energy_emissions"] = (
                emissions_electricity
            )
            outputs["data:environmental_impact:operational:emissions"] = (
                emissions_electricity + emissions_fuel
            )
            outputs["data:environmental_impact:operational:emission_factor"] = (
                (emissions_electricity + emissions_fuel) / op_range / payload
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mission_option = self.options["mission"]

        if mission_option == "design" or mission_option == "both":
            fuel_consumed = inputs["data:mission:sizing:fuel"]
            electricity_consumed = inputs["data:mission:sizing:energy"]

            design_range = inputs["data:TLAR:range"]
            payload = inputs["data:weight:aircraft:payload"]

            partials[
                "data:environmental_impact:sizing:fuel_emissions", "data:mission:sizing:fuel"
            ] = self.energy_content_fuel * self.carbon_intensity_fuel
            partials["data:environmental_impact:sizing:emissions", "data:mission:sizing:fuel"] = (
                self.energy_content_fuel * self.carbon_intensity_fuel
            )

            partials[
                "data:environmental_impact:sizing:energy_emissions", "data:mission:sizing:energy"
            ] = self.carbon_intensity_electricity * KWH_TO_MJ
            partials["data:environmental_impact:sizing:emissions", "data:mission:sizing:energy"] = (
                self.carbon_intensity_electricity * KWH_TO_MJ
            )

            partials[
                "data:environmental_impact:sizing:emission_factor", "data:mission:sizing:energy"
            ] = self.carbon_intensity_electricity * KWH_TO_MJ / design_range / payload
            partials[
                "data:environmental_impact:sizing:emission_factor", "data:mission:sizing:fuel"
            ] = self.energy_content_fuel * self.carbon_intensity_fuel / design_range / payload
            partials["data:environmental_impact:sizing:emission_factor", "data:TLAR:range"] = (
                -(
                    electricity_consumed * self.carbon_intensity_electricity * KWH_TO_MJ
                    + self.energy_content_fuel * self.carbon_intensity_fuel * fuel_consumed
                )
                / design_range**2.0
                / payload
            )
            partials[
                "data:environmental_impact:sizing:emission_factor", "data:weight:aircraft:payload"
            ] = (
                -(
                    electricity_consumed * self.carbon_intensity_electricity * KWH_TO_MJ
                    + self.energy_content_fuel * self.carbon_intensity_fuel * fuel_consumed
                )
                / design_range
                / payload**2.0
            )

        if mission_option == "operational" or mission_option == "both":
            fuel_consumed = inputs["data:mission:operational:fuel"]
            electricity_consumed = inputs["data:mission:operational:energy"]

            op_range = inputs["data:mission:operational:range"]
            payload = inputs["data:mission:operational:payload:mass"]

            partials[
                "data:environmental_impact:operational:fuel_emissions",
                "data:mission:operational:fuel",
            ] = self.energy_content_fuel * self.carbon_intensity_fuel
            partials[
                "data:environmental_impact:operational:emissions", "data:mission:operational:fuel"
            ] = self.energy_content_fuel * self.carbon_intensity_fuel

            partials[
                "data:environmental_impact:operational:energy_emissions",
                "data:mission:operational:energy",
            ] = self.carbon_intensity_electricity * KWH_TO_MJ
            partials[
                "data:environmental_impact:operational:emissions",
                "data:mission:operational:energy",
            ] = self.carbon_intensity_electricity * KWH_TO_MJ

            partials[
                "data:environmental_impact:operational:emission_factor",
                "data:mission:operational:energy",
            ] = self.carbon_intensity_electricity * KWH_TO_MJ / op_range / payload
            partials[
                "data:environmental_impact:operational:emission_factor",
                "data:mission:operational:fuel",
            ] = self.energy_content_fuel * self.carbon_intensity_fuel / op_range / payload
            partials[
                "data:environmental_impact:operational:emission_factor",
                "data:mission:operational:range",
            ] = (
                -(
                    electricity_consumed * self.carbon_intensity_electricity * KWH_TO_MJ
                    + self.energy_content_fuel * self.carbon_intensity_fuel * fuel_consumed
                )
                / op_range**2.0
                / payload
            )
            partials[
                "data:environmental_impact:operational:emission_factor",
                "data:mission:operational:payload:mass",
            ] = (
                -(
                    electricity_consumed * self.carbon_intensity_electricity * KWH_TO_MJ
                    + self.energy_content_fuel * self.carbon_intensity_fuel * fuel_consumed
                )
                / op_range
                / payload**2.0
            )

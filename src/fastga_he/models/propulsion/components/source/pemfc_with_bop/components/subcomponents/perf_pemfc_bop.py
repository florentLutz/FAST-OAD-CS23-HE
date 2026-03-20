# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .compressor import PerformancesCompressor
from .inlet import PerformancesInlet
from .pump import PerformancesPump
from .heat_exchanger import PerformancesHeatExchanger
from .humidifier import PerformancesHumidifier
from .pipe import PerformancesPipe
from .pump import PerformancesPump
from .nozzle import PerformancesNozzle
from .diffuser import PerformancesDiffuser
from .perf_pemf_bop_primary_hex_properties import PerformancesPrimaryHeatExchangerThermalBalance
from .perf_pemf_bop_supplement_hex_properties import (
    PerformancesSupplementHeatExchangerThermalBalance,
)
from .perf_pemf_bop_air_inlet_flow import PerformancesAirInletAirMassFlow
from .fluid_characteristics import FluidSpecificHeatCapacity


class PerformancesPEMFCBOP(om.Group):
    """
    Group to compute the performances of the PEMFC BOP.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "coolant_fluid_type",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )
        self.options.declare(
            name="compressor_id",
            default=None,
            desc="Identifier of the compressor",
            allow_none=False,
        )
        self.options.declare(
            name="coolant_component_ids",
            default="None",
            desc="A list of the TBS components that use coolant",
            allow_none=False,
        )
        self.options.declare(
            name="pipe_id",
            default=None,
            desc="Identifier of the pipe",
            allow_none=False,
        )
        self.options.declare(
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air inlet",
            allow_none=False,
        )
        self.options.declare(
            name="primary_heat_exchanger_id",
            default=None,
            desc="Identifier of the primary heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            name="supplement_heat_exchanger_id",
            default=None,
            desc="Identifier of the supplement heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            name="humidifier_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )
        self.options.declare(
            name="diffuser_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )
        self.options.declare(
            name="nozzle_id",
            default=None,
            desc="Identifier of the nozzle",
            allow_none=False,
        )
        self.options.declare(
            name="pump_id",
            default=None,
            desc="Identifier of the pump",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        compress_id = self.options["compressor_id"]
        coolant_component_ids = self.options["coolant_component_ids"]
        pipe_id = self.options["pipe_id"]
        air_inlet_id = self.options["air_inlet_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]
        valve_id = self.options["valve_id"]
        diffuser_id = self.options["diffuser_id"]
        nozzle_id = self.options["nozzle_id"]
        pump_id = self.options["pump_id"]
        humidifier_id = self.options["humidifier_id"]

        self.add_subsystem(
            "compressor",
            PerformancesCompressor(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compress_id,
                number_of_points=number_of_points,
                connected_humidifier_id=humidifier_id,
                connected_heat_exchanger_ids=primary_heat_exchanger_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "humidifier",
            PerformancesHumidifier(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                humidifier_id=humidifier_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "primary_heat_exchanger_air_properties",
            PerformancesPrimaryHeatExchangerThermalBalance(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=primary_heat_exchanger_id,
                coolant_fluid_type=coolant_fluid_type,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "primary_heat_exchanger",
            PerformancesHeatExchanger(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=primary_heat_exchanger_id,
                coolant_fluid_type=coolant_fluid_type,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inlet_air_flow_rate",
            PerformancesAirInletAirMassFlow(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                air_inlet_id=air_inlet_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inlet",
            PerformancesInlet(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                air_inlet_id=air_inlet_id,
                number_of_points=number_of_points,
                supplied_heat_exchanger_ids=[
                    primary_heat_exchanger_id,
                    supplement_heat_exchanger_id,
                ],
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "diffuser",
            PerformancesDiffuser(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "supplement_heat_exchanger_properties",
            PerformancesSupplementHeatExchangerThermalBalance(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=supplement_heat_exchanger_id,
                coolant_fluid_type=coolant_fluid_type,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "supplement_heat_exchanger",
            PerformancesHeatExchanger(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=supplement_heat_exchanger_id,
                coolant_fluid_type=coolant_fluid_type,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )

        self.connect(
            "primary_heat_exchanger_air_properties.air_inlet_temperature",
            "primary_heat_exchanger.air_inlet_temperature",
        )
        self.connect(
            "primary_heat_exchanger_air_properties.air_outlet_temperature",
            "primary_heat_exchanger.air_outlet_temperature",
        )
        self.connect(
            "primary_heat_exchanger_air_properties.air_static_pressure",
            "primary_heat_exchanger.air_static_pressure",
        )
        self.connect(
            "primary_heat_exchanger_air_properties.coolant_inlet_temperature",
            "primary_heat_exchanger.coolant_inlet_temperature",
        )
        self.connect(
            "primary_heat_exchanger_air_properties.coolant_outlet_temperature",
            "primary_heat_exchanger.coolant_outlet_temperature",
        )
        self.connect("inlet_air_flow_rate.air_mass_flow", "inlet.air_mass_flow")
        self.connect(
            "inlet.throat_total_pressure",
            "diffuser.throat_air_pressure",
        )
        self.connect(
            "inlet.throat_total_temperature",
            "diffuser.throat_air_temperature",
        )

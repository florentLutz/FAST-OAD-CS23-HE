# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .compressor import PerformancesCompressor
from .inlet import PerformancesInlet
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
from .perf_pemf_bop_speed_of_sound import PerformancesAirSpeedOfSound
from .perf_pemf_bop_mach import PerformancesAirMach


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
        pipe_id = self.options["pipe_id"]
        air_inlet_id = self.options["air_inlet_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]
        diffuser_id = self.options["diffuser_id"]
        nozzle_id = self.options["nozzle_id"]
        pump_id = self.options["pump_id"]
        humidifier_id = self.options["humidifier_id"]

        self.add_subsystem(
            "primary_heat_exchanger_loop",
            PerformancesPrimaryHeatExchangerLoop(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                coolant_fluid_type=coolant_fluid_type,
                compressor_id=compress_id,
                primary_heat_exchanger_id=primary_heat_exchanger_id,
                humidifier_id=humidifier_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "altitude", "exterior_temperature", "air_consumption"],
        )
        self.add_subsystem(
            "speed_of_sound",
            PerformancesAirSpeedOfSound(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            "mach", PerformancesAirMach(number_of_points=number_of_points), promotes=["*"]
        )
        self.add_subsystem(
            "air_inlet",
            PerformancesInlet(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                air_inlet_id=air_inlet_id,
                number_of_points=number_of_points,
            ),
            promotes=[
                "data:*",
                "mach",
                "exterior_temperature",
                "altitude",
                "true_airspeed",
                "density",
                "air_consumption",
            ],
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
            "supplement_heat_exchanger_air_properties",
            PerformancesSupplementHeatExchangerThermalBalance(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                supplement_heat_exchanger_id=supplement_heat_exchanger_id,
                connected_air_inlet_id=air_inlet_id,
                coolant_fluid_type=coolant_fluid_type,
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
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
        self.add_subsystem(
            "nozzle",
            PerformancesNozzle(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                number_of_points=number_of_points,
                connected_heat_exchanger_id=supplement_heat_exchanger_id,
            ),
            promotes=["data:*", "exterior_temperature", "true_airspeed"],
        )
        self.add_subsystem(
            "pipe",
            PerformancesPipe(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                pipe_id=pipe_id,
                coolant_fluid_type=coolant_fluid_type,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pump",
            PerformancesPump(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                pump_id=pump_id,
                coolant_fluid_type=coolant_fluid_type,
                coolant_component_ids=[
                    pipe_id,
                    primary_heat_exchanger_id,
                    supplement_heat_exchanger_id,
                ],
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "bop_drag",
            PerformancesBOPDrag(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                drag_component_ids=[air_inlet_id, nozzle_id],
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "bop_power",
            PerformancesBOPPower(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compress_id,
                pump_id=pump_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )

        self.connect(
            "supplement_heat_exchanger_air_properties.air_inlet_temperature",
            "supplement_heat_exchanger.air_inlet_temperature",
        )
        self.connect(
            "supplement_heat_exchanger_air_properties.air_outlet_temperature",
            "supplement_heat_exchanger.air_outlet_temperature",
        )
        self.connect(
            "supplement_heat_exchanger_air_properties.air_static_pressure",
            "supplement_heat_exchanger.air_static_pressure",
        )
        self.connect(
            "supplement_heat_exchanger_air_properties.coolant_inlet_temperature",
            "supplement_heat_exchanger.coolant_inlet_temperature",
        )
        self.connect(
            "supplement_heat_exchanger_air_properties.coolant_outlet_temperature",
            "supplement_heat_exchanger.coolant_outlet_temperature",
        )
        self.connect("diffuser.diffuser_exit_total_pressure", "nozzle.diffuser_exit_pressure")
        self.connect("diffuser.diffuser_exit_total_temperature", "nozzle.diffuser_exit_temperature")
        self.connect(
            "diffuser.diffuser_exit_total_pressure",
            "supplement_heat_exchanger_air_properties.diffuser_exit_total_pressure",
        )
        self.connect(
            "diffuser.diffuser_exit_total_temperature",
            "supplement_heat_exchanger_air_properties.diffuser_exit_total_temperature",
        )
        self.connect("diffuser.exit_air_speed", "nozzle.entry_air_speed")
        self.connect("air_inlet.inlet_air_mass_flow", "nozzle.air_mass_flow_rate")
        self.connect("air_inlet.throat_total_pressure", "diffuser.throat_air_pressure")
        self.connect("air_inlet.throat_total_temperature", "diffuser.throat_air_temperature")
        self.connect("air_inlet.throat_air_speed", "diffuser.throat_air_speed")
        self.connect(
            "air_inlet.ambient_pressure",
            "supplement_heat_exchanger_air_properties.ambient_pressure",
        )


class PerformancesBOPDrag(om.ExplicitComponent):
    """
    Computes the mass of the PEMFC BOP by summing the mass of each component.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack BOP",
            allow_none=False,
        )
        self.options.declare(
            name="drag_component_ids",
            default="None",
            desc="A list of the TBS components that induce drag",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        drag_component_ids = self.options["drag_component_ids"]

        for component_id in drag_component_ids:
            self.add_input(
                name="data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + component_id
                + ":drag",
                units="N",
                val=np.nan,
                shape=number_of_points,
            )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_drag",
            units="N",
            val=0.0,
            shape=number_of_points - 2,
        )

    def setup_partials(self):
        self.declare_partials(
            "*",
            "*",
            val=1.0,
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        drag_component_ids = self.options["drag_component_ids"]

        bop_drag = np.zeros(number_of_points)

        for component_id in drag_component_ids:
            bop_drag += inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + component_id
                + ":drag"
            ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_drag"
        ] = bop_drag[1:-1]


class PerformancesBOPPower(om.ExplicitComponent):
    """
    Computes the mass of the PEMFC BOP by summing the mass of each component.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack BOP",
            allow_none=False,
        )
        self.options.declare(
            name="pump_id",
            default="None",
            desc="Identifier of the pump in the TMS",
            allow_none=False,
        )
        self.options.declare(
            name="compressor_id",
            default="None",
            desc="Identifier of the compressor in the TMS",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        pump_id = self.options["pump_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            units="kW",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            units="kW",
            val=10.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        pump_id = self.options["pump_id"]

        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            "*",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        pump_id = self.options["pump_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required"
        ] = np.clip(
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + compressor_id
                + ":power_required"
            ]
            + inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pump_id
                + ":power_rating"
            ],
            0.0,
            240.0,
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        pump_id = self.options["pump_id"]

        unclipped_power_required = (
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + compressor_id
                + ":power_required"
            ]
            + inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + pump_id
                + ":power_rating"
            ]
        )

        clipped_power_required = np.clip(unclipped_power_required, 0.0, 240.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required",
        ] = np.where(unclipped_power_required == clipped_power_required, 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + pump_id
            + ":power_rating",
        ] = np.where(unclipped_power_required == clipped_power_required, 1.0, 0.0)


class PerformancesPrimaryHeatExchangerLoop(om.Group):
    """
    Group to compute the performances of the primary heat exchanger loop of the PEMFC BOP.
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
            name="primary_heat_exchanger_id",
            default=None,
            desc="Identifier of the primary heat exchanger",
            allow_none=False,
        )
        self.options.declare(
            name="humidifier_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        compress_id = self.options["compressor_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]
        humidifier_id = self.options["humidifier_id"]

        self.add_subsystem(
            "compressor",
            PerformancesCompressor(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compress_id,
                number_of_points=number_of_points,
                connected_humidifier_id=humidifier_id,
                connected_heat_exchanger_id=primary_heat_exchanger_id,
            ),
            promotes=["data:*", "altitude", "exterior_temperature", "air_consumption"],
        )
        self.add_subsystem(
            "humidifier",
            PerformancesHumidifier(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                humidifier_id=humidifier_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "air_consumption"],
        )
        self.add_subsystem(
            "primary_heat_exchanger_air_properties",
            PerformancesPrimaryHeatExchangerThermalBalance(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                primary_heat_exchanger_id=primary_heat_exchanger_id,
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

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 5
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

        self.connect(
            "compressor.compressor_outlet_temperature",
            "primary_heat_exchanger_air_properties.compressor_outlet_temperature",
        )
        self.connect(
            "compressor.compressor_pressure_supply",
            "primary_heat_exchanger_air_properties.compressor_pressure_supply",
        )
        self.connect(
            "humidifier.oxidizer_temperature",
            "primary_heat_exchanger_air_properties.oxidizer_temperature",
        )
        self.connect(
            "humidifier.oxidizer_pressure",
            "primary_heat_exchanger_air_properties.oxidizer_pressure",
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

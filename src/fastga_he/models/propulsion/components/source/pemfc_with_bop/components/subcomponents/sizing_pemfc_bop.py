# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .compressor import SizingCompressorWeight
from .coolant_tank import SizingCoolantTank
from .diffuser import SizingDiffuser
from .heat_exchanger import SizingHeatExchanger
from .humidifier import SizingHumidifier
from .inlet import SizingInlet
from .nozzle import SizingNozzle
from .pipe import SizingPipe
from .pump import SizingPumpWeight
from .valve import SizingValve


class SizingPEMFCBOP(om.Group):
    """
    Group to compute the dimensions of the PEMFC BOP.
    """

    def initialize(self):
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
            name="valve_id",
            default=None,
            desc="Identifier of the valve",
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
        self.options.declare(
            name="coolant_tank_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        pipe_id = self.options["pipe_id"]
        air_inlet_id = self.options["air_inlet_id"]
        supplement_heat_exchanger_id = self.options["supplement_heat_exchanger_id"]
        primary_heat_exchanger_id = self.options["primary_heat_exchanger_id"]
        valve_id = self.options["valve_id"]
        diffuser_id = self.options["diffuser_id"]
        nozzle_id = self.options["nozzle_id"]
        pump_id = self.options["pump_id"]
        coolant_tank_id = self.options["coolant_tank_id"]
        coolant_fluid_type = self.options["coolant_fluid_type"]
        humidifier_id = self.options["humidifier_id"]
        sizing_component_ids = [
            compressor_id,
            air_inlet_id,
            diffuser_id,
            primary_heat_exchanger_id,
            supplement_heat_exchanger_id,
            valve_id,
            nozzle_id,
            pump_id,
            coolant_tank_id,
            pipe_id,
            humidifier_id,
        ]
        coolant_component_ids = [primary_heat_exchanger_id, supplement_heat_exchanger_id, pipe_id]

        self.add_subsystem(
            "compressor",
            SizingCompressorWeight(
                compressor_id=compressor_id, pemfc_stack_bop_id=pemfc_stack_bop_id
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "air_inlet",
            SizingInlet(
                air_inlet_id=air_inlet_id,
                pemfc_stack_bop_id=pemfc_stack_bop_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "diffuser",
            SizingDiffuser(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                diffuser_id=diffuser_id,
                connected_heat_exchanger_id=supplement_heat_exchanger_id,
                connected_air_inlet_id=air_inlet_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "primary_heat_exchanger",
            SizingHeatExchanger(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=primary_heat_exchanger_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "supplement_heat_exchanger",
            SizingHeatExchanger(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                heat_exchanger_id=supplement_heat_exchanger_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "humidifier",
            SizingHumidifier(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                humidifier_id=humidifier_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pipe",
            SizingPipe(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                pipe_id=pipe_id,
                coolant_fluid_type=coolant_fluid_type,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "coolant_tank",
            SizingCoolantTank(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                coolant_component_ids=coolant_component_ids,
                coolant_tank_id=coolant_tank_id,
                coolant_fluid_type=coolant_fluid_type,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "valve",
            SizingValve(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                valve_id=valve_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "nozzle",
            SizingNozzle(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                nozzle_id=nozzle_id,
                connected_heat_exchanger_id=supplement_heat_exchanger_id,
                connected_diffuser_id=diffuser_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "pump",
            SizingPumpWeight(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                pump_id=pump_id,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "bop_mass",
            SizingBOPMass(
                pemfc_stack_bop_id=pemfc_stack_bop_id, sizing_component_ids=sizing_component_ids
            ),
            promotes=["data:*"],
        )


class SizingBOPMass(om.ExplicitComponent):
    """
    Computes the mass of the PEMFC BOP by summing the mass of each component.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack BOP",
            allow_none=False,
        )
        self.options.declare(
            name="sizing_component_ids",
            default="None",
            desc="A list of the TBS components that are in the sizing group",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        sizing_component_ids = self.options["sizing_component_ids"]

        for component_id in sizing_component_ids:
            self.add_input(
                name="data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + component_id
                + ":mass",
                units="kg",
                val=np.nan,
            )

        self.add_output(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_mass",
            units="kg",
            val=10.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        sizing_component_ids = self.options["sizing_component_ids"]
        bop_mass = 0.0

        for component_id in sizing_component_ids:
            bop_mass += inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + component_id
                + ":mass"
            ]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:" + pemfc_stack_bop_id + ":bop_mass"
        ] = np.clip(bop_mass, 0.0, 1000.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        sizing_component_ids = self.options["sizing_component_ids"]
        bop_mass = 0.0

        for component_id in sizing_component_ids:
            bop_mass += inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + component_id
                + ":mass"
            ]

        clipped_bop_mass = np.clip(bop_mass, 0.0, 1000.0)

        for component_id in sizing_component_ids:
            partials[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":bop_mass",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + component_id
                + ":mass",
            ] = np.where(bop_mass == clipped_bop_mass, 1.0, 1e-6)

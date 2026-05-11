# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .compressor import SizingCompressorWeight
from .humidifier import SizingHumidifier
from .finned_heat_sink import SizingFinnedHeatSink


class SizingPEMFCBOPSimplified(om.Group):
    """
    Group to compute the dimensions of the simplified PEMFC BOP.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="compressor_id",
            default=None,
            desc="Identifier of the compressor",
            allow_none=False,
        )
        self.options.declare(
            name="humidifier_id",
            default=None,
            desc="Identifier of the humidifier",
            allow_none=False,
        )
        self.options.declare(
            name="finned_heat_sink_id",
            default=None,
            desc="Identifier of the finned heat sink",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]
        humidifier_id = self.options["humidifier_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

        sizing_component_ids = [
            compressor_id,
            humidifier_id,
            finned_heat_sink_id,
        ]

        self.add_subsystem(
            "compressor",
            SizingCompressorWeight(
                compressor_id=compressor_id, pemfc_stack_bop_id=pemfc_stack_bop_id
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
            "finned_heat_sink",
            SizingFinnedHeatSink(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
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
        self.declare_partials("*", "*", val=1.0)

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
        ] = bop_mass

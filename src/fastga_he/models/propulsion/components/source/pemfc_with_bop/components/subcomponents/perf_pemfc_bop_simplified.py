# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .compressor import PerformancesCompressor
from .humidifier import PerformancesHumidifier
from .finned_heat_sink import PerformancesFinnedHeatSink
from .perf_pemf_compresssed_air_heat import PerformancesCompressedAirHeat


class PerformancesPEMFCBOPSimplified(om.Group):
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
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compress_id = self.options["compressor_id"]
        humidifier_id = self.options["humidifier_id"]
        finned_heat_sink_id = self.options["finned_heat_sink_id"]

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
            "compressor",
            PerformancesCompressor(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compress_id,
                number_of_points=number_of_points,
                connected_humidifier_id=humidifier_id,
                connected_heat_exchanger_id="primary_heat_exchanger_1",
            ),
            promotes=["data:*", "altitude", "exterior_temperature", "air_consumption"],
        )
        self.add_subsystem(
            "compressed_air_heat",
            PerformancesCompressedAirHeat(
                number_of_points=number_of_points, pemfc_stack_bop_id=pemfc_stack_bop_id
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "finned_heat_sink",
            PerformancesFinnedHeatSink(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                finned_heat_sink_id=finned_heat_sink_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "altitude", "exterior_temperature", "true_airspeed"],
        )
        self.add_subsystem(
            "bop_power",
            PerformancesBOPPower(
                pemfc_stack_bop_id=pemfc_stack_bop_id,
                compressor_id=compress_id,
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )

        self.connect(
            "compressor.compressor_outlet_temperature",
            "compressed_air_heat.compressor_outlet_temperature",
        )
        self.connect(
            "compressor.compressor_pressure_supply",
            "compressed_air_heat.compressor_pressure_supply",
        )
        self.connect(
            "humidifier.oxidizer_temperature",
            "compressed_air_heat.oxidizer_temperature",
        )
        self.connect(
            "humidifier.oxidizer_pressure",
            "compressed_air_heat.oxidizer_pressure",
        )


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
            name="compressor_id",
            default="None",
            desc="Identifier of the compressor in the TMS",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

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
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        compressor_id = self.options["compressor_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required"
        ] = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + compressor_id
            + ":power_required"
        ]

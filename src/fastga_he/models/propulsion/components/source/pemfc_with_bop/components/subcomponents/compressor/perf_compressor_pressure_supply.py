# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCompressorPressureSupply(om.ExplicitComponent):
    """
    Computation of the amount of pressure supplied by the compressor.
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
            name="connected_humidifier_id",
            default=None,
            desc="Identifier of the connected humidifier",
            allow_none=False,
        )
        self.options.declare(
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        humidifier_id = self.options["connected_humidifier_id"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        self.add_input(
            "compressor_pressure_target",
            units="Pa",
            val=np.nan,
            desc="Input anode pressure if applicable",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + humidifier_id
            + ":air_pressure_drop",
            val=1e4,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            val=1e4,
            units="Pa",
            shape=number_of_points,
        )

        self.add_output(
            "compressor_pressure_supply",
            val=0.3,
            units="Pa",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["compressor_pressure_supply"] = np.sum(inputs.value())

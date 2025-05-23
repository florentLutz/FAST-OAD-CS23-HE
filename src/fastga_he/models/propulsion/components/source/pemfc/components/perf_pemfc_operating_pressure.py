# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import DEFAULT_PRESSURE


class PerformancesPEMFCStackOperatingPressure(om.ExplicitComponent):
    """
    Operating pressure computation of the PEMFC stack.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="compressor_connection",
            default=False,
            types=bool,
            desc="The PEMFC stack operation pressure have to adjust based on compressor "
            "connection for the oxygen/air inlet",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        compressor_connection = self.options["compressor_connection"]

        if compressor_connection:
            self.add_input(
                "input_pressure",
                units="Pa",
                val=np.full(number_of_points, np.nan),
                desc="Input pressure from the compressor if applicable",
            )
        else:
            self.add_input("ambient_pressure", units="Pa", val=np.full(number_of_points, np.nan))

        self.add_output(
            name="operating_pressure",
            units="Pa",
            val=np.full(number_of_points, DEFAULT_PRESSURE),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        compressor_connection = self.options["compressor_connection"]
        if compressor_connection:
            outputs["operating_pressure"] = inputs["input_pressure"]
        else:
            outputs["operating_pressure"] = inputs["ambient_pressure"]

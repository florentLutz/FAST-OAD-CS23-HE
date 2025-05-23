# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualDepreciation(om.ExplicitComponent):
    """
    Computation of the aircraft depreciation per year.
    """

    def setup(self):
        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            "data:cost:operation:depreciation_rate",
            val=0.9,
            desc="The portion of value that depreciates over the service life",
        )
        self.add_input(
            name="data:TLAR:aircraft_lifespan",
            val=20.0,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )

        self.add_output(
            "data:cost:operation:annual_depreciation_cost",
            val=15000.0,
            units="USD/yr",
            desc="Annual depreciation cost of the aircraft",
        )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_depreciation_cost"] = (
            inputs["data:cost:msp_per_unit"]
            * inputs["data:cost:operation:depreciation_rate"]
            / inputs["data:TLAR:aircraft_lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:operation:annual_depreciation_cost", "data:cost:msp_per_unit"] = (
            inputs["data:cost:operation:depreciation_rate"] / inputs["data:TLAR:aircraft_lifespan"]
        )

        partials[
            "data:cost:operation:annual_depreciation_cost", "data:cost:operation:depreciation_rate"
        ] = inputs["data:cost:msp_per_unit"] / inputs["data:TLAR:aircraft_lifespan"]

        partials["data:cost:operation:annual_depreciation_cost", "data:TLAR:aircraft_lifespan"] = (
            -inputs["data:cost:msp_per_unit"]
            * inputs["data:cost:operation:depreciation_rate"]
            / inputs["data:TLAR:aircraft_lifespan"] ** 2.0
        )

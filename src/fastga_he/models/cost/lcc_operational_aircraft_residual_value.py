# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalAircraftResidualValue(om.ExplicitComponent):
    """
    Computation of the aircraft residual value based on the flat depreciation method given by
    :cite:`iata:2016' for both AirFrance-KLM and Lufthansa.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            "data:cost:operation:annual_depreciation_cost",
            val=15000.0,
            units="USD/yr",
            desc="Annual depreciation cost of the aircraft",
        )

        self.add_output(
            "data:cost:operation:residual_value",
            val=1e5,
            units="USD",
            shape=years_of_service + 1,
            desc="Residual value of the aircraft at across the lifespan",
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(years_of_service + 1),
            cols=np.zeros(years_of_service + 1),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_service = self.options["years_of_service"]

        outputs["data:cost:operation:residual_value"] = inputs["data:cost:msp_per_unit"] - inputs[
            "data:cost:operation:annual_depreciation_cost"
        ] * np.arange(years_of_service + 1)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        partials["data:cost:operation:residual_value", "data:cost:msp_per_unit"] = np.ones(
            years_of_service + 1
        )

        partials[
            "data:cost:operation:residual_value", "data:cost:operation:annual_depreciation_cost"
        ] = -np.arange(years_of_service + 1)

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTurboshaftInFlightH2OEmissions(om.ExplicitComponent):
    """
    Computation of the turboshaft in flight water vapour emissions, will be based on a simple emissions
    index whose default value is taken from :cite:`kim:2005`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        turboshaft_id = self.options["turboshaft_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("fuel_consumed_t", np.full(number_of_points, np.nan), units="kg")
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":emission_index:H2O",
            units="g/kg",
            val=1237.0,
        )

        self.add_output("H2O_emissions", np.full(number_of_points, 1237.0), units="g")

        self.declare_partials(
            of="H2O_emissions",
            wrt="fuel_consumed_t",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="H2O_emissions",
            wrt="data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":emission_index:H2O",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        outputs["H2O_emissions"] = (
            inputs["fuel_consumed_t"]
            * inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":emission_index:H2O"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        partials["H2O_emissions", "fuel_consumed_t"] = np.full_like(
            inputs["fuel_consumed_t"],
            inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":emission_index:H2O"
            ],
        )
        partials[
            "H2O_emissions",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":emission_index:H2O",
        ] = inputs["fuel_consumed_t"]
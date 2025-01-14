# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

SPECIES_LIST = ["CO2", "CO", "NOx", "SOx", "HC", "H2O"]


class PerformancesGeneratorInFlightEmissionsSum(om.ExplicitComponent):
    """
    Addition of the emissions of all pollutants at each step of the flight. Will be zero for
    generator. The burden will come from the thermal machines connected to it. However, since
    generators are a source, they must output emissions but they will be 0.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the generator", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        generator_id = self.options["generator_id"]

        for specie in SPECIES_LIST:
            self.add_input(specie + "_emissions", val=np.full(number_of_points, np.nan), units="g")
            # For the LCA module we will adopt the following nomenclature:
            # "LCA" + phase + component + pollutant
            self.add_output(
                "data:environmental_impact:operation:sizing:he_power_train:generator:"
                + generator_id
                + ":"
                + specie,
                units="g",
                val=3.1e5,
            )
            self.declare_partials(
                of="data:environmental_impact:operation:sizing:he_power_train:generator:"
                + generator_id
                + ":"
                + specie,
                wrt=specie + "_emissions",
                rows=np.zeros(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        generator_id = self.options["generator_id"]

        for specie in SPECIES_LIST:
            outputs[
                "data:environmental_impact:operation:sizing:he_power_train:generator:"
                + generator_id
                + ":"
                + specie
            ] = np.sum(inputs[specie + "_emissions"])

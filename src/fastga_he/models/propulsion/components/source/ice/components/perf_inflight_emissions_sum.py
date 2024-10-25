# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

SPECIES_LIST = ["CO2", "CO", "NOx", "SOx", "HC", "H2O", "lead"]


class PerformancesICEInFlightEmissionsSum(om.ExplicitComponent):
    """
    Addition of the emissions of all pollutants at each step of the flight.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        ice_id = self.options["ice_id"]

        for specie in SPECIES_LIST:
            self.add_input(specie + "_emissions", val=np.full(number_of_points, np.nan), units="g")
            # For the LCA module we will adopt the following nomenclature:
            # "LCA" + phase + component + pollutant
            self.add_output(
                "data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie, units="g", val=0.0
            )
            self.declare_partials(
                of="data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie,
                wrt=specie + "_emissions",
                rows=np.zeros(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]

        for specie in SPECIES_LIST:
            outputs["data:LCA:operation:he_power_train:ICE:" + ice_id + ":" + specie] = np.sum(
                inputs[specie + "_emissions"]
            )

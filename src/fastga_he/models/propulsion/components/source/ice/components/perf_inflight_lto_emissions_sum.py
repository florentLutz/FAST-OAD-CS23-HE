# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .perf_inflight_emissions_sum import SPECIES_LIST

THRESHOLD_ALTITUDE_LTO = 3000.0  # In ft


class PerformancesICELTOEmissionsSum(om.ExplicitComponent):
    """
    Addition of the emissions of all pollutants for steps of the flight in the LTO cycle. The
    threshold altitude's value is 3000ft as taken from :cite:`brooker:2006`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lto_mask = None

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

        self.add_input("altitude", val=np.full(number_of_points, np.nan), units="ft")

        for specie in SPECIES_LIST:
            self.add_input(specie + "_emissions", val=np.full(number_of_points, np.nan), units="g")
            # For the LCA module we will adopt the following nomenclature:
            # "LCA" + phase + component + pollutant
            self.add_output(
                "data:environmental_impact:operation:sizing:he_power_train:ICE:"
                + ice_id
                + ":"
                + specie
                + "_LTO",
                units="g",
                val=0.0,
            )
            self.declare_partials(
                of="data:environmental_impact:operation:sizing:he_power_train:ICE:"
                + ice_id
                + ":"
                + specie
                + "_LTO",
                wrt=specie + "_emissions",
                rows=np.zeros(number_of_points),
                cols=np.arange(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]
        self.lto_mask = np.where(inputs["altitude"] < THRESHOLD_ALTITUDE_LTO, 1.0, 0.0)

        for specie in SPECIES_LIST:
            outputs[
                "data:environmental_impact:operation:sizing:he_power_train:ICE:"
                + ice_id
                + ":"
                + specie
                + "_LTO"
            ] = np.sum(inputs[specie + "_emissions"] * self.lto_mask)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ice_id = self.options["ice_id"]

        for specie in SPECIES_LIST:
            partials[
                "data:environmental_impact:operation:sizing:he_power_train:ICE:"
                + ice_id
                + ":"
                + specie
                + "_LTO",
                specie + "_emissions",
            ] = self.lto_mask

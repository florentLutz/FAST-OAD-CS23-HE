# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

SPECIES_LIST = ["CO2", "CO", "NOx", "SOx", "HC", "H2O"]


class PerformancesPEMFCStackInFlightEmissionsSum(om.ExplicitComponent):
    """
    Addition of the emissions of all pollutants at each step of the flight. Will be zero for
    PEMFC stack for almost all species but still added for consistency with turboshaft and
    battery_pack.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Default is set as None, so it is not computed when not wanted. In the mission, it
        # will be enabled
        self.options.declare(
            "number_of_points_reserve",
            default=None,
            desc="number of equilibrium to be treated in reserve",
            types=int,
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        for specie in SPECIES_LIST:
            self.add_input(specie + "_emissions", val=np.full(number_of_points, np.nan), units="g")
            # For the LCA module we will adopt the following nomenclature:
            # "LCA" + phase + component + pollutant
            self.add_output(
                "data:environmental_impact:operation:sizing:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + specie,
                units="g",
                val=3.1e5,
            )

            if number_of_points_reserve:
                self.add_output(
                    "data:environmental_impact:operation:sizing:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":"
                    + specie
                    + "_main_route",
                    units="g",
                    val=0.0,
                    desc="Emission of "
                    + specie
                    + " excluding reserve, quantity of interest for the LCA",
                )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        for specie in SPECIES_LIST:
            self.declare_partials(
                of="data:environmental_impact:operation:sizing:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + specie,
                wrt=specie + "_emissions",
                rows=np.zeros(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

            if number_of_points_reserve:
                val_partial = np.ones(number_of_points)
                val_partial[-number_of_points_reserve - 1 : -1] = np.zeros(number_of_points_reserve)

                self.declare_partials(
                    of="data:environmental_impact:operation:sizing:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":"
                    + specie
                    + "_main_route",
                    wrt=specie + "_emissions",
                    rows=np.zeros(number_of_points),
                    cols=np.arange(number_of_points),
                    val=val_partial,
                )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        for specie in SPECIES_LIST:
            outputs[
                "data:environmental_impact:operation:sizing:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":"
                + specie
            ] = np.sum(inputs[specie + "_emissions"])

            if number_of_points_reserve:
                outputs[
                    "data:environmental_impact:operation:sizing:he_power_train:PEMFC_stack:"
                    + pemfc_stack_id
                    + ":"
                    + specie
                    + "_main_route"
                ] = np.sum(inputs[specie + "_emissions"]) - np.sum(
                    inputs[specie + "_emissions"][-number_of_points_reserve - 1 : -1]
                )

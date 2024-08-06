# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBatteryModuleWeight(om.ExplicitComponent):
    """
    Computation of the weight of one of the battery module. Based on the weight of the individual
    cell and an overhead coefficient to account for packaging, cooling, etc ... This
    simplified methodology which we are going to use for now is taken from zhao:2018 with the
    overhead taken at 0.5.
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="cell_weight_ref",
            types=float,
            default=50.0e-3,
            desc="Weight of the reference cell for the battery construction [kg]",
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
            val=np.nan,
            desc="Number of cells in series inside one battery module",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell_weight_fraction",
            val=0.5,
            desc="Cell weight fraction, represents the contribution to the battery mass of the "
            "cells",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:mass",
            units="kg",
            val=100.0,
            desc="Mass of one module of the battery",
        )
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:mass",
            val=self.options["cell_weight_ref"],
            units="g",
            desc="Mass of the cell used for the assembly of the battery pack",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:mass",
            wrt="*",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:mass",
            wrt=[],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:mass"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * self.options["cell_weight_ref"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_weight_fraction"
            ]
        )

        outputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:mass"] = (
            self.options["cell_weight_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:mass",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = (
            self.options["cell_weight_ref"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_weight_fraction"
            ]
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:mass",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell_weight_fraction",
        ] = -(
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * self.options["cell_weight_ref"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_weight_fraction"
            ]
            ** 2.0
        )

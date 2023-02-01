# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBatteryModuleVolume(om.ExplicitComponent):
    """
    Computation of the volume of one of the battery module. Based on the volume of the individual
    cell and an overhead coefficient to account for packaging, cooling, etc ... A simple
    coefficient will be used to take into account the volume of the overhead.
    """

    def initialize(self):

        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="cell_volume_ref",
            types=float,
            default=17.63e-3,
            desc="Volume of the reference cell for the battery construction [L]",
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
            + ":cell_volume_fraction",
            val=0.8,
            desc="Cell volume fraction, represents the contribution to the battery volume of the "
            "cells",
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume",
            units="L",
            val=50.0,
            desc="Volume of one module of the battery",
        )
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:volume",
            val=self.options["cell_volume_ref"],
            units="L",
            desc="Volume of the cell used for the assembly of the battery pack",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume",
            wrt="*",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:volume",
            wrt=[],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume"
        ] = (
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * self.options["cell_volume_ref"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_volume_fraction"
            ]
        )

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:volume"
        ] = self.options["cell_volume_ref"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = (
            self.options["cell_volume_ref"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_volume_fraction"
            ]
        )
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":module:volume",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell_volume_fraction",
        ] = -(
            inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * self.options["cell_volume_ref"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell_volume_fraction"
            ]
            ** 2.0
        )

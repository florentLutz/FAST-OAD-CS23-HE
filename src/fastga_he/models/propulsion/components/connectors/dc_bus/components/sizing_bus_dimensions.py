# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBusBarDimensions(om.ExplicitComponent):
    """
    Computation of the bus bar dimensions, assumes two conducting plates surrounded by an
    insulation layer whose thickness is computed based on PD criterion.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
            units="m",
            val=0.3,
            desc="Length of the bus bar conductor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":height",
            units="m",
            val=10e-3,
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":width",
            units="m",
            val=20e-3,
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":length",
            units="m",
            val=0.3,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":height",
            wrt=[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            ],
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":width",
            wrt=[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            ],
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":length",
            wrt=[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        # One plate conducting and one plate for return of current
        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":height"] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness"
            ]
            + 3.0
            * inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness"]
        )

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":width"] = (
            inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width"]
            + 2.0
            * inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness"]
        )

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":length"] = (
            inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length"]
            + 2.0
            * inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":height",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
        ] = 2.0
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":height",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
        ] = 3.0

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":width",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:width",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":width",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
        ] = 2.0

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":length",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:length",
        ] = 1.0
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":length",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
        ] = 2.0

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesPowerDensity(om.ExplicitComponent):
    """
    Power density of the auxiliary load based on the mass specific heat rejection and the heat
    to power ratio.
    """

    def initialize(self):
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):
        aux_load_id = self.options["aux_load_id"]

        self.add_input(
            name="data:propulsion:he_power_train:aux_load:"
            + aux_load_id
            + ":mass_specific_heat_rejection",
            val=np.nan,
            units="kW/kg",
            desc="The waste heat of PEMFC to TMS power ratio",
        )
        self.add_input(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio",
            val=np.nan,
            desc="The waste heat of PEMFC to TMS power ratio",
        )

        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density",
            units="kW/kg",
            val=0.5,
            desc="Power density of the auxiliary load",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density"] = (
            inputs[
                "data:propulsion:he_power_train:aux_load:"
                + aux_load_id
                + ":mass_specific_heat_rejection"
            ]
            * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio",
        ] = inputs[
            "data:propulsion:he_power_train:aux_load:"
            + aux_load_id
            + ":mass_specific_heat_rejection"
        ]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density",
            "data:propulsion:he_power_train:aux_load:"
            + aux_load_id
            + ":mass_specific_heat_rejection",
        ] = inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio"]

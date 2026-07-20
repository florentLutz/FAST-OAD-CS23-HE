# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import scipy as sp
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesHeatPowerRatio(om.ExplicitComponent):
    """
    Computation of the ratio between the TMS power and the PEMFC waste heat.
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
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max",
            val=np.nan,
            units="kW",
            desc="Heat from PEMFC to dissipate",
        )

        self.add_output(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio",
            val=0.2,
            desc="The waste heat of PEMFC to TMS power ratio",
        )

    def setup_partials(self):
        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio"] = (
            0.0767
            * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max"]
            ** 0.114
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":heat_power_ratio",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max",
        ] = (
            0.0087438
            * inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max"]
            ** -0.886
        )

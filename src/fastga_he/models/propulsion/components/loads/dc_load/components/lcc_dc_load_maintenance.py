# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDCLoadMaintenance(om.ExplicitComponent):
    """
    Computation of the maintenance cost of the DC loads including the electronics of the powertrain.
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
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_unit",
            units="USD",
            val=np.nan,
            desc="Cost of the DC load per unit including electronics",
        )

        self.add_input(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":lifespan",
            units="yr",
            val=15.0,
            desc="Expected lifetime of the electric component, typically around 15 year",
        )

        self.add_output(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":maintenance_per_unit",
            units="USD/yr",
            val=100.0,
            desc="Annual maintenance cost of the electronics",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":maintenance_per_unit"
        ] = (
            inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_unit"]
            / inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":lifespan"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":maintenance_per_unit",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_unit",
        ] = 1.0 / inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":lifespan"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":maintenance_per_unit",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":lifespan",
        ] = (
            -inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cost_per_unit"]
            / inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":lifespan"] ** 2.0
        )

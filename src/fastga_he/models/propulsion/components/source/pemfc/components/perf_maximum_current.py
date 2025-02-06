# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

MAX_DEFAULT_STACK_CURRENT = 1000.0
MIN_DEFAULT_STACK_CURRENT = 100.0


class PerformancesMaximumCurrent(om.ExplicitComponent):
    """
    Class to identify the maximum current output from PEMFC.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_min",
            units="A",
            val=MIN_DEFAULT_STACK_CURRENT,
            desc="Minimum current to the pemfc during the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max",
            units="A",
            val=MAX_DEFAULT_STACK_CURRENT,
            desc="Maximum current to the pemfc during the mission",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max"] = (
            np.nanmax(inputs["dc_current_out"])
        )

        outputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_min"] = (
            np.nanmin(inputs["dc_current_out"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max",
            "dc_current_out",
        ] = np.where(inputs["dc_current_out"] == np.max(inputs["dc_current_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_min",
            "dc_current_out",
        ] = np.where(inputs["dc_current_out"] == np.min(inputs["dc_current_out"]), 1.0, 0.0)

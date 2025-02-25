# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_DEFAULT_STACK_POWER, MAX_DEFAULT_STACK_CURRENT


class PerformancesPEMFCStackMaximum(om.ExplicitComponent):
    """
    Computation that identifies the maximum power and current output from PEMFC.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )


    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("power_out", units="kW", val=np.full(number_of_points, np.nan))

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=MAX_DEFAULT_STACK_POWER,
            desc="Maximum power to PEMFC during the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            units="A",
            val=MAX_DEFAULT_STACK_CURRENT,
            desc="Maximum current to PEMFC during the mission",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            wrt="power_out",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            wrt="dc_current_out",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"] = (
            np.max(inputs["power_out"])
        )

        outputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max"] = (
            np.max(inputs["dc_current_out"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            "power_out",
        ] = np.where(inputs["power_out"] == np.max(inputs["power_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            "dc_current_out",
        ] = np.where(inputs["dc_current_out"] == np.max(inputs["dc_current_out"]), 1.0, 0.0)

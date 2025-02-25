# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_CURRENT_DENSITY_EMPIRICAL, MAX_CURRENT_DENSITY_ANALYTICAL


class PerformancesPEMFCStackCurrentDensity(om.ExplicitComponent):
    """
    Computation of the current density, simply based on the current and effective area.
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
        self.options.declare(
            name="model_fidelity",
            default="empirical",
            desc="Select the polarization model between empirical and analytical. The "
            "Aerostak 200W empirical polarization model is set as default.",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]
        model_fidelity = self.options["model_fidelity"]

        if model_fidelity == "analytical":
            max_current_density = MAX_CURRENT_DENSITY_ANALYTICAL
        else:
            max_current_density = MAX_CURRENT_DENSITY_EMPIRICAL

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=np.nan,
            desc="Effective fuel cell area in the stack",
        )

        self.add_output(
            "fc_current_density",
            val=np.full(number_of_points, max_current_density),
            units="A/cm**2",
            desc="Current density of PEMFC stack",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="dc_current_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs["fc_current_density"] = (
            inputs["dc_current_out"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        partials["fc_current_density", "dc_current_out"] = np.full(
            number_of_points,
            inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
            * -1,
        )

        partials[
            "fc_current_density",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
        ] = (
            -inputs["dc_current_out"]
            / inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
            ** 2
        )

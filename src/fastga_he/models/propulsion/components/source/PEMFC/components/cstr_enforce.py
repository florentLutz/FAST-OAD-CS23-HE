# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA
] = "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce"

MAX_CURRENT_DENSITY = 0.7  # A/cm^2


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce",
)
class ConstraintsEffectiveAreaEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power seen by the pemfc during the mission is used for
    the sizing.
    """

    def initialize(self):

        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stacks",
            allow_none=False,
        )

    def setup(self):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max",
            units="A",
            val=np.nan,
            desc="Maximum current the PEMFC stack has be given during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=16.8,
            desc="Effective area of PEMFC chemical reaction",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max",
            val=1 / MAX_CURRENT_DENSITY,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
        ] = (
            inputs["data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max"]
            / MAX_CURRENT_DENSITY
        )

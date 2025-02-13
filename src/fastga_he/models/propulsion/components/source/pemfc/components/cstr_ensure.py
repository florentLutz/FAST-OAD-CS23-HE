# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA

MAX_CURRENT_DENSITY = 0.7  # A/cm^2


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.ensure",
)
class ConstraintsEffectiveAreaEnsure(om.ExplicitComponent):
    """
    Class that ensures that the maximum power seen by the PEMFC stack during the mission is below the
    one used for sizing, ensuring each component works below its minimum.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
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

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=np.nan,
            desc="Effective area of PEMFC chemical reaction",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":effective_area",
            units="cm**2",
            val=-0.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":effective_area",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area",
            val=-1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":effective_area",
            wrt="data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max",
            val=1 / MAX_CURRENT_DENSITY,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "constraints:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":effective_area"
        ] = (
            -inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":effective_area"
            ]
            + inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":current_max"
            ]
            / MAX_CURRENT_DENSITY
        )

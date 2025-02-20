# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.ensure",
)
class ConstraintsPEMFCStackEffectiveAreaEnsure(om.ExplicitComponent):
    """
    Class that ensures that the maximum current seen by the PEMFC stack during the mission is below
    the one used for sizing, ensuring each component works below its maximum. This is achieved by
    adjusting the PEMFC effective area, which is the area of polymer electrolyte membrane.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        max_current_density = self.options["max_current_density"]
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            units="A",
            val=np.nan,
            desc="Maximum current the PEMFC stack has to provide during mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=np.nan,
            desc="Effective area of PEMFC's polymer electrolyte membrane",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":effective_area",
            units="cm**2",
            val=-0.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":effective_area",
            wrt="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            val=-1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":effective_area",
            wrt="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            val=1 / max_current_density,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        max_current_density = self.options["max_current_density"]
        outputs[
            "constraints:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":effective_area"
        ] = (
            inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max"]
            / max_current_density
            - inputs[
                "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
            ]
        )

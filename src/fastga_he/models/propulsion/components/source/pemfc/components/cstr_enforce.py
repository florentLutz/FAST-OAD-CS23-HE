# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA] = (
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce",
)
class ConstraintsPEMFCStackEffectiveAreaEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the PEMFC during the mission is used for
    the sizing. The effective area is defined by dividing the maximum current with maximum
    current density of PEMFC.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the pemfc stack",
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

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=16.8,
            desc="Effective area of PEMFC's polymer electrolyte membrane",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            wrt="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            val=1 / max_current_density,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        max_current_density = self.options["max_current_density"]
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
        ] = (
            inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max"]
            / max_current_density
        )

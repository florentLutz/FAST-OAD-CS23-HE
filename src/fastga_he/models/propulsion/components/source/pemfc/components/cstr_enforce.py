# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import (
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA,
    SUBMODEL_CONSTRAINTS_PEMFC_POWER,
    MAX_CURRENT_DENSITY_EMPIRICAL,
    MAX_CURRENT_DENSITY_ANALYTICAL,
)

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA] = (
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_PEMFC_POWER] = (
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.power.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_EFFECTIVE_AREA,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.effective_area.enforce",
)
class ConstraintsPEMFCStackEffectiveAreaEnforce(om.ExplicitComponent):
    """
    Class that enforcing the maximum current seen by the PEMFC stack during the mission is used for
    sizing. The effective area is defined by dividing the maximum current with the maximum
    current density of the PEMFC stack.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="model_fidelity",
            default="empirical",
            desc="Select the polarization model between empirical and analytical. The "
            "Aerostak 200W empirical polarization model is set as default.",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        model_fidelity = self.options["model_fidelity"]

        if model_fidelity == "analytical":
            max_current_density = MAX_CURRENT_DENSITY_ANALYTICAL
        else:
            max_current_density = MAX_CURRENT_DENSITY_EMPIRICAL

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            units="A",
            val=np.nan,
            desc="Maximum current that the PEMFC stack has to provide during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            units="cm**2",
            val=16.8,
            desc="Effective area of the PEMFC's polymer electrolyte membrane",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area",
            wrt="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max",
            val=1.0 / max_current_density,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        model_fidelity = self.options["model_fidelity"]

        if model_fidelity == "analytical":
            max_current_density = MAX_CURRENT_DENSITY_ANALYTICAL
        else:
            max_current_density = MAX_CURRENT_DENSITY_EMPIRICAL

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":effective_area"
        ] = (
            inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":current_max"]
            / max_current_density
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_PEMFC_POWER,
    "fastga_he.submodel.propulsion.constraints.pemfc_stack.power.enforce",
)
class ConstraintsPEMFCStackPowerEnforce(om.ExplicitComponent):
    """
    Class that enforcing the maximum power seen by the PEMFC stack during the mission is used for
    sizing.
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
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power that the PEMFC stack has to provide during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_rating",
            units="kW",
            val=16.8,
            desc="Maximum power that the PEMFC stack can supply continuously",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_rating",
            wrt="data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_rating"
        ] = inputs["data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":power_max"]

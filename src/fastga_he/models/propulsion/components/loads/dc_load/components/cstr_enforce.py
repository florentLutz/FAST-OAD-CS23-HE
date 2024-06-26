# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_AUX_LOAD_POWER,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_AUX_LOAD_POWER
] = "fastga_he.submodel.propulsion.constraints.aux_load.power.enforce"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_AUX_LOAD_POWER,
    "fastga_he.submodel.propulsion.constraints.aux_load.power.enforce",
)
class ConstraintsPowerEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power requested by the aux load during the mission is
    used for the sizing, ensuring a fitted design for the power of each component.
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
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum value of the power the auxiliary load requests",
        )

        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating",
            units="kW",
            val=250.0,
            desc="Max continuous power of the auxiliary load",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating",
            wrt="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        aux_load_id = self.options["aux_load_id"]

        outputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating"
        ] = inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max"]

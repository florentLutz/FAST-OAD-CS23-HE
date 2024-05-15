# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_GEARBOX_TORQUE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_GEARBOX_TORQUE
] = "fastga_he.submodel.propulsion.constraints.gearbox.torque.enforce"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_GEARBOX_TORQUE,
    "fastga_he.submodel.propulsion.constraints.gearbox.torque.enforce",
)
class ConstraintsTorqueEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum torque seen by the gearbox during the mission is
    used for the sizing, ensuring a fitted design for the torque of each component.
    """

    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )

    def setup(self):

        gearbox_id = self.options["gearbox_id"]

        self.add_input(
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the output torque of the gearbox",
        )

        self.add_output(
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating",
            units="N*m",
            val=250.0,
            desc="Max continuous output torque of the gearbox",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating",
            wrt="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        gearbox_id = self.options["gearbox_id"]

        outputs[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating"
        ] = inputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max"]

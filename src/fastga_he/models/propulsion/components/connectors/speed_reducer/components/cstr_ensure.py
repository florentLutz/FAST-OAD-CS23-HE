# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_SPEED_REDUCER_TORQUE,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_SPEED_REDUCER_TORQUE,
    "fastga_he.submodel.propulsion.constraints.speed_reducer.torque.ensure",
)
class ConstraintsTorqueEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum torque seen by the speed reducer during the
    mission and the value used for sizing, ensuring each component works below its maximum.
    """

    def initialize(self):
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]

        self.add_input(
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_in_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the input torque of the gearbox",
        )
        self.add_input(
            "data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous input torque of the gearbox",
        )
        self.add_input(
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_out_max",
            units="N*m",
            val=np.nan,
            desc="Maximum value of the output torque of the gearbox",
        )
        self.add_input(
            "data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous output torque of the gearbox",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_rating",
            units="N*m",
            val=250.0,
            desc="Respected if <0",
        )
        self.add_output(
            "constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_rating",
            units="N*m",
            val=250.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_rating",
            wrt="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_rating",
            wrt="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_rating",
            val=-1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_rating",
            wrt="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_rating",
            wrt="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_rating",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        outputs[
            "constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_rating"
        ] = (
            inputs[
                "data:propulsion:he_power_train:speed_reducer:"
                + speed_reducer_id
                + ":torque_in_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:speed_reducer:"
                + speed_reducer_id
                + ":torque_in_rating"
            ]
        )
        outputs[
            "constraints:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_rating"
        ] = (
            inputs[
                "data:propulsion:he_power_train:speed_reducer:"
                + speed_reducer_id
                + ":torque_out_max"
            ]
            - inputs[
                "data:propulsion:he_power_train:speed_reducer:"
                + speed_reducer_id
                + ":torque_out_rating"
            ]
        )

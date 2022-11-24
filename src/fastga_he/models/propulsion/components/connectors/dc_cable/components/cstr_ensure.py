# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_DC_LINE

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_LINE, "fastga_he.submodel.propulsion.constraints.dc_line.ensure"
)
class ConstraintsEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum seen by the DC line during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):

        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
            val=np.nan,
            units="V",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            val=np.nan,
            units="A",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            val=np.nan,
            units="V",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber",
            val=np.nan,
            units="A",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            units="V",
            val=-50.0,
            desc="Ensure the line voltage rating is enough for the maximum it sees during the "
            "mission, constraint is respected if < 0",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber",
            units="A",
            val=-50.0,
            desc="Ensure the line current rating is enough for the maximum it sees during the "
            "mission, constraint is respected if < 0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":voltage_caliber",
            ],
            method="exact",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":current_caliber",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        outputs[
            "constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max"]
            - inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber"
            ]
        )
        outputs[
            "constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max"]
            - inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_caliber"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        partials[
            "constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber",
        ] = -1.0
        partials[
            "constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_caliber",
        ] = -1.0

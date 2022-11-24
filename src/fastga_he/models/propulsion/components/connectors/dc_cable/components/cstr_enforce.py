# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_DC_LINE

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_DC_LINE
] = "fastga_he.submodel.propulsion.constraints.dc_line.enforce"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_LINE, "fastga_he.submodel.propulsion.constraints.dc_line.enforce"
)
class ConstraintsEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum seen by the DC line during the mission are used for the
    sizing, ensuring a fitted design of each component.
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

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            val=800.0,
            units="V",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":current_caliber",
            val=500.0,
            units="A",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_caliber",
            wrt="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber"
        ] = inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max"]
        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_caliber"
        ] = inputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max"]

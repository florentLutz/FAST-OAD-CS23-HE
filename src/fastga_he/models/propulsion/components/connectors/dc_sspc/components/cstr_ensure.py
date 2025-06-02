# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC, SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC,
    "fastga_he.submodel.propulsion.constraints.dc_sspc.current.ensure",
)
class ConstraintsCurrentEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum current seen by the SSPC during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the SSPC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            val=np.nan,
            units="A",
            desc="Maximum current flowing through the SSPC",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            units="A",
            val=-100.0,
            desc="Ensure the SSPC current rating is enough for the maximum it sees during the "
            "mission, constraint is respected if < 0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        outputs[
            "constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max"]
            - inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"]
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC,
    "fastga_he.submodel.propulsion.constraints.dc_sspc.voltage.ensure",
)
class ConstraintsVoltageEnsure(om.ExplicitComponent):
    """
    Class that computes the difference between the maximum voltage seen by the SSPC during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the SSPC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
            val=np.nan,
            units="V",
            desc="Maximum voltage flowing through the SSPC",
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            units="V",
            val=-100.0,
            desc="Ensure the SSPC voltage rating is enough for the maximum it sees during the "
            "mission, constraint is respected if < 0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        outputs[
            "constraints:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max"]
            - inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"]
        )

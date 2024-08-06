# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC, SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC] = (
    "fastga_he.submodel.propulsion.constraints.dc_sspc.current.enforce"
)
oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC] = (
    "fastga_he.submodel.propulsion.constraints.dc_sspc.voltage.enforce"
)

# If value of sizing current and voltage are too low, like in the case of an inactive SSPC,
# these value should be chosen
MINIMUM_CURRENT = 1.0
MINIMUM_VOLTAGE = 10.0


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_CURRENT_DC_SSPC,
    "fastga_he.submodel.propulsion.constraints.dc_sspc.current.enforce",
)
class ConstraintsCurrentEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the DC SSPC during the mission is used
    for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            val=np.nan,
            units="A",
            desc="Maximum current flowing through the SSPC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            units="A",
            val=800.0,
            desc="Current caliber of the SSPC",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        current_max = max(
            MINIMUM_CURRENT,
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max"],
        )

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"] = (
            current_max
        )


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SSPC,
    "fastga_he.submodel.propulsion.constraints.dc_sspc.voltage.enforce",
)
class ConstraintsVoltageEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum voltage seen by the DC SSPC during the mission is used
    for the sizing, ensuring a fitted design of each component.
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
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
            val=np.nan,
            units="V",
            desc="Maximum voltage flowing through the SSPC",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            units="V",
            val=777.0,
            desc="Voltage caliber of the SSPC",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        voltage_max = max(
            MINIMUM_VOLTAGE,
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max"],
        )

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"] = (
            voltage_max
        )

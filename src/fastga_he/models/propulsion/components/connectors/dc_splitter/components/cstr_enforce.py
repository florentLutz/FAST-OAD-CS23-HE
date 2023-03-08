# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import (
    SUBMODEL_CONSTRAINTS_CURRENT_DC_SPLITTER,
    SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SPLITTER,
)

import openmdao.api as om
import numpy as np

import fastoad.api as oad

oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_CURRENT_DC_SPLITTER
] = "fastga_he.submodel.propulsion.constraints.dc_splitter.current.enforce"
oad.RegisterSubmodel.active_models[
    SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SPLITTER
] = "fastga_he.submodel.propulsion.constraints.dc_splitter.voltage.enforce"


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_CURRENT_DC_SPLITTER,
    "fastga_he.submodel.propulsion.constraints.dc_splitter.current.enforce",
)
class ConstraintsCurrentEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum current seen by the DC splitter during the mission is used
    for the sizing, ensuring a fitted design of each component.
    """

    def initialize(self):

        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            units="A",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":current_caliber",
            units="A",
            val=500.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_caliber",
            wrt="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_caliber"
        ] = inputs["data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max"]


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_VOLTAGE_DC_SPLITTER,
    "fastga_he.submodel.propulsion.constraints.dc_splitter.voltage.enforce",
)
class ConstraintsVoltageEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum voltage seen by the DC splitter during the mission is used
    for the sizing, ensuring a fitted design of each component.
    """

    def initialize(self):

        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max",
            units="V",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:"
            + dc_splitter_id
            + ":voltage_caliber",
            units="V",
            val=800.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_caliber",
            wrt="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_caliber"
        ] = inputs["data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max"]

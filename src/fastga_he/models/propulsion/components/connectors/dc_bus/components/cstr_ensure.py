# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_DC_BUS

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_DC_BUS, "fastga_he.submodel.propulsion.constraints.dc_bus.enforce"
)
class ConstraintsEnsure(om.ExplicitComponent):
    """
    Class that computes the differetence between the maximum seen by the DC bus during the
    mission and the value used for sizing, ensuring each component works below its maxima.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
            units="A",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max",
            units="V",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            units="V",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
            units="A",
            val=np.nan,
        )

        self.add_output(
            name="constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            units="V",
            val=800.0,
            desc="Ensure the bus voltage rating is enough for the maximum it sees during the "
            "mission, constraint is respected if < 0",
        )
        self.add_output(
            name="constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
            units="A",
            val=500.0,
            desc="Ensure the bus current rating is enough for the maximum it sees during the "
            "mission, constraint is respected if < 0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max",
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            ],
            method="exact",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
            wrt=[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        outputs[
            "constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max"]
            - inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber"]
        )
        outputs[
            "constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber"
        ] = (
            inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max"]
            - inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        partials[
            "constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
        ] = -1.0
        partials[
            "constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
        ] = 1.0
        partials[
            "constraints:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_caliber",
        ] = -1.0

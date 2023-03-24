# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_ICE_SL_POWER

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_ICE_SL_POWER,
    "fastga_he.submodel.propulsion.constraints.ice.sea_level_power.ensure",
)
class ConstraintsSeaLevelPowerEnsure(om.ExplicitComponent):
    """
    Class that ensures that the maximum power seen by the motor during the mission is below the
    one used for sizing, ensuring each component works below its minimum.
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_max_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor has to provide at Sea Level",
        )
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            units="kW",
            val=-0.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":power_max_SL",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        outputs["constraints:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"] = (
            inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_max_SL"]
            - inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"]
        )

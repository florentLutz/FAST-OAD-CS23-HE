# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER] = (
    "fastga_he.submodel.propulsion.constraints.turboshaft.rated_power.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER,
    "fastga_he.submodel.propulsion.constraints.turboshaft.rated_power.enforce",
)
class ConstraintsRatedPowerEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power seen by the turboshaft during the mission is used for
    the sizing as the rated power, ensuring a fitted design.
    """

    def initialize(self):
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
            units="kW",
            val=np.nan,
            desc="Maximum power the turboshaft has to provide",
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=600.0,
            desc="Maximum power the turboshaft can provide",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            wrt="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        outputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"] = (
            inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max"]
        )

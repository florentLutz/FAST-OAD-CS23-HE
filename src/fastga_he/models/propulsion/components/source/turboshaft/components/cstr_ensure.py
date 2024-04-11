# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER,
    "fastga_he.submodel.propulsion.constraints.turboshaft.rated_power.enforce",
)
class ConstraintsRatedPowerEnsure(om.ExplicitComponent):
    """
    Class that ensures that the maximum power seen by the turboshaft during the mission is below the
    one used for sizing, ensuring each component works below its minimum.
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
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Maximum power the turboshaft can provide",
        )

        self.add_output(
            "constraints:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=-0.0,
            desc="Respected if <0",
        )

        self.declare_partials(
            of="constraints:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":power_rating",
            wrt="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
            val=1.0,
        )
        self.declare_partials(
            of="constraints:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":power_rating",
            wrt="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        outputs[
            "constraints:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ] = (
            inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max"]
            - inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"]
        )

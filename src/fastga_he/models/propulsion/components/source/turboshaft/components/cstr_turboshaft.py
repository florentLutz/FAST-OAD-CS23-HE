# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER


class ConstraintsTurboshaft(om.Group):
    """
    Class that gather the different constraints for the turboshaft be they ensure or enforce.
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

        option_turboshaft_id = {"turboshaft_id": turboshaft_id}

        self.add_subsystem(
            name="constraints_SL_power",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_TURBOSHAFT_RATED_POWER, options=option_turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_for_power_rate",
            subsys=ConstraintTurboshaftPowerRateMission(turboshaft_id=turboshaft_id),
            promotes=["*"],
        )


class ConstraintTurboshaftPowerRateMission(om.ExplicitComponent):
    """
    This class will define the value of the maximum power we use to get the power rate inside the
    mission, it is mandatory that we compute it outside the mission when sizing the power train
    or else when recomputing the wing area it will be stuck at one which we don't want. Also the
    turboshaft will be like the ICE.
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
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Maximum power the turboshaft can provide",
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":shaft_power_rating",
            units="kW",
            val=42000.0,
            desc="Value of the maximum power the turboshaft can provide used for power rate",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":shaft_power_rating"
        ] = inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"]

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_CONSTRAINTS_HIGH_RPM_ICE_SL_POWER


class ConstraintsHighRPMICE(om.Group):
    """
    Class that gather the different constraints for the high RPM ICE be they ensure or enforce.
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        option_high_rpm_ice_id = {"high_rpm_ice_id": high_rpm_ice_id}

        self.add_subsystem(
            name="constraints_SL_power",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_CONSTRAINTS_HIGH_RPM_ICE_SL_POWER, options=option_high_rpm_ice_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_for_power_rate",
            subsys=ConstraintHighRPMICEPowerRateMission(high_rpm_ice_id=high_rpm_ice_id),
            promotes=["*"],
        )


class ConstraintHighRPMICEPowerRateMission(om.ExplicitComponent):
    """
    This class will define the value of the maximum power we use to get the power rate inside the
    mission, it is mandatory that we compute it outside the mission when sizing the power train
    or else when recomputing the wing area it will be stuck at one which we don't want. Also the
    ICE will be different from the generator and PMSM because the max shaft power is actually
    used for the sizing which is different because they use torque
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":shaft_power_rating",
            units="kW",
            val=42000.0,
            desc="Value of the maximum power the motor can provide at Sea Level used for sizing",
        )

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":shaft_power_rating"
        ] = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL"
        ]

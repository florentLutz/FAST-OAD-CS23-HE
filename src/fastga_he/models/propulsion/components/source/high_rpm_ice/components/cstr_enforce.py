# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import SUBMODEL_CONSTRAINTS_HIGH_RPM_ICE_SL_POWER

import fastoad.api as oad

oad.RegisterSubmodel.active_models[SUBMODEL_CONSTRAINTS_HIGH_RPM_ICE_SL_POWER] = (
    "fastga_he.submodel.propulsion.constraints.high_rpm_ice.sea_level_power.enforce"
)


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_HIGH_RPM_ICE_SL_POWER,
    "fastga_he.submodel.propulsion.constraints.high_rpm_ice.sea_level_power.enforce",
)
class ConstraintsSeaLevelPowerEnforce(om.ExplicitComponent):
    """
    Class that enforces that the maximum power seen by the motor during the mission is used for
    the sizing, ensuring a fitted design for the torque of each component.
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
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL",
            units="kW",
            val=np.nan,
            desc="Maximum power the motor has to provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL",
            units="kW",
            val=250.0,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":power_rating_SL",
            wrt="data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_rating_SL"
        ] = inputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL"
        ]

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum sea level equivalent power to use for the ICE sizing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("equivalent_SL_power", units="W", val=np.full(number_of_points, np.nan))
        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL",
            units="W",
            val=250e3,
            desc="Maximum power the motor has to provide at Sea Level",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL",
            wrt="equivalent_SL_power",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":shaft_power_max",
            units="W",
            val=42000.0,
            desc="Maximum power seen during the mission, without accounting for altitude effect, "
            "only used for power rate",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:high_rpm_ICE:"
            + high_rpm_ice_id
            + ":shaft_power_max",
            wrt="shaft_power_out",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL"
        ] = np.max(inputs["equivalent_SL_power"])

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":shaft_power_max"
        ] = np.max(inputs["shaft_power_out"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":power_max_SL",
            "equivalent_SL_power",
        ] = np.where(
            inputs["equivalent_SL_power"] == np.max(inputs["equivalent_SL_power"]), 1.0, 0.0
        )

        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":shaft_power_max",
            "shaft_power_out",
        ] = np.where(inputs["shaft_power_out"] == np.max(inputs["shaft_power_out"]), 1.0, 0.0)

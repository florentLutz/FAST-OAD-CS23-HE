# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesSFC(om.ExplicitComponent):
    """
    Computation of the ICE sfc for the required MEP. For more information on the method check
    in ...ice_rotax.methodology.fuel_consumption_regression.py
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":sfc",
            units="g/kW/h",
            val=np.nan,
        )

        self.add_output(
            "specific_fuel_consumption", units="g/kW/h", val=300.0, shape=number_of_points
        )

        self.declare_partials(
            of="specific_fuel_consumption",
            wrt=["data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":sfc"],
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]
        number_of_points = self.options["number_of_points"]

        outputs["specific_fuel_consumption"] = np.full(
            number_of_points,
            inputs["data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":sfc"],
        )

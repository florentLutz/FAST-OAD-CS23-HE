# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTemperatureConstant(om.ExplicitComponent):
    """
    Computation of assigning conductor temperature for resistance calculation.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":conductor_temperature_mission",
            val=np.nan,
            units="degK",
            desc="Performances of the conductor will be computed assuming it has a constant "
            "temperature",
        )

        self.add_output(
            "winding_temperature",
            val=np.full(number_of_points, 293.15),
            units="degK",
            desc="temperature of the conductor winding",
            shape=number_of_points,
            lower=1.0,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        outputs["winding_temperature"] = np.full(
            number_of_points,
            inputs[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":conductor_temperature_mission"
            ],
        )

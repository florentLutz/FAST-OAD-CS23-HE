# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMeanEffectivePressure(om.ExplicitComponent):
    """
    Computation of the ICE mean effective pressure for the required torque, formula is based on
    :cite:`lutz:2022`
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        ice_id = self.options["ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            units="m**3",
            val=np.nan,
        )
        self.add_input("data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number", val=4.0)
        self.add_input("torque_out", units="N*m", val=np.nan, shape=number_of_points)

        self.add_output(
            "mean_effective_pressure", units="bar", val=15.0, shape=number_of_points, lower=0.0
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        volume = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume"]
        strokes_nb = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number"]
        torque_out = inputs["torque_out"]

        mep = 2.0 * np.pi * strokes_nb * torque_out / volume

        # To convert in bar
        outputs["mean_effective_pressure"] = mep * 1e-5

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]
        number_of_points = self.options["number_of_points"]

        volume = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume"]
        strokes_nb = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number"]
        torque_out = inputs["torque_out"]

        partials[
            "mean_effective_pressure",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
        ] = (
            -(2.0 * np.pi * strokes_nb * torque_out / volume ** 2.0) * 1e-5
        )
        partials[
            "mean_effective_pressure",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number",
        ] = (
            2.0 * np.pi * torque_out / volume * 1e-5
        )
        partials["mean_effective_pressure", "torque_out"] = (
            np.eye(number_of_points) * 2.0 * np.pi * strokes_nb / volume * 1e-5
        )

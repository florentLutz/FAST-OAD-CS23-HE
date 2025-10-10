# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import logging

_LOGGER = logging.getLogger(__name__)


class PerformancesElectromagneticFactor(om.ExplicitComponent):
    """
    Computation of preventing the output RPM exceeds the centrifugal force limit RPM and the
    first bending mode resonance RPM.
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
            name="flux_geometry_factor",
            val=np.nan,
            shape=number_of_points,
            desc="The ratio between the actual flux density and its upper limit",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_factor",
            desc="Remain at 1.0 if the maximum air gap flux density is within the range",
            val=1.0,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_factor",
            wrt="flux_geometry_factor",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        flux_factor = (
            np.max(inputs["flux_geometry_factor"])
            if np.max(inputs["flux_geometry_factor"]) > 1.0
            else 1.0
        )

        if np.max(inputs["flux_geometry_factor"]) > 1.0:
            _LOGGER.info(
                msg="Maximum design air gap flux density exceeded. Increase rotor diameter to stay "
                "within acceptable range."
            )

        outputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_factor"
        ] = np.sqrt(flux_factor)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        flux_factor = (
            np.max(inputs["flux_geometry_factor"])
            if np.max(inputs["flux_geometry_factor"]) > 1.0
            else 1.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_factor",
            "flux_geometry_factor",
        ] = np.where(
            (flux_factor > 1.0) & (inputs["flux_geometry_factor"] == flux_factor),
            0.5 / np.sqrt(flux_factor),
            0.0,
        )

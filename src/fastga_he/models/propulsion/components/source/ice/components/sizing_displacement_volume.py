# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingICEDisplacementVolume(om.ExplicitComponent):
    """
    Computation of the ICE displacement volume, based on technological hypothesis on the
    maximum Mean Effective Pressure in the cylinder and the maximum rpm. The number of strokes
    can be changed but we recommend leaving it at 4, see :cite:`lutz:2022`
    """

    def initialize(self):
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):
        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":pme_max",
            units="bar",
            val=20.62,
            desc="Maximum Mean Effective Pressure in the ICE cylinders",
        )
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":rpm_max",
            units="min**-1",
            val=2700.0,
            desc="Maximum RPM on the shaft of the engine",
        )
        self.add_input("data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number", val=4.0)
        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            units="W",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            units="m**3",
            val=0.0107,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ice_id = self.options["ice_id"]

        pme_max = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":pme_max"] * 1e5  # In Pa
        rpm_max = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":rpm_max"]
        omega_max = rpm_max * 2.0 * np.pi / 60.0
        max_power = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"]
        strokes_nb = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number"]

        volume = (max_power * 2.0 * np.pi * strokes_nb) / (pme_max * omega_max)

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume"] = volume

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ice_id = self.options["ice_id"]

        pme_max = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":pme_max"]
        rpm_max = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":rpm_max"]
        omega_max = rpm_max * 2.0 * np.pi / 60.0
        max_power = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"]
        strokes_nb = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":pme_max",
        ] = -(max_power * 2.0 * np.pi * strokes_nb) / (pme_max**2.0 * 1e5 * omega_max)
        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":rpm_max",
        ] = -(max_power * 2.0 * np.pi * strokes_nb) / (
            pme_max * 1e5 * rpm_max**2.0 * 2.0 * np.pi / 60.0
        )
        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
        ] = (2.0 * np.pi * strokes_nb) / (pme_max * 1e5 * omega_max)
        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":displacement_volume",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":strokes_number",
        ] = (max_power * 2.0 * np.pi) / (pme_max * 1e5 * omega_max)

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesSurfaceCurrentDensity(om.ExplicitComponent):
    """
    Computation of the  current density of the SM PMSM. The formula is obtained from
    equation (II-38) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_in_one_phase",
            units="A",
            val=np.full(number_of_points, np.nan),
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_per_slot",
            val=np.nan,
            desc="Number of wire per stator slot",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
            val=np.nan,
            desc="Number of conductor slots on the motor stator",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            units="m",
            desc="Stator bore diameter of the SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
            val=np.nan,
            desc="The factor considers the cable winding effect in the stator slot",
        )

        self.add_output(
            name="surface_current_density",
            val=1.1e5,
            units="A/m",
            shape=number_of_points,
            desc="The surface current density of the winding conductor cable",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=["ac_current_rms_in_one_phase"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_per_slot",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        i_rms = inputs["ac_current_rms_in_one_phase"]
        num_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]
        num_wire = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_per_slot"]
        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        k_winding = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor"]

        outputs["surface_current_density"] = (
            np.sqrt(2.0) * k_winding * num_slot * num_wire * i_rms / (np.pi * bore_diameter)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        i_rms = inputs["ac_current_rms_in_one_phase"]
        num_slot = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number"
        ]
        num_wire = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_per_slot"]
        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        k_winding = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor"]

        partials[
            "surface_current_density",
            "ac_current_rms_in_one_phase",
        ] = (
            np.full(number_of_points, np.sqrt(2.0))
            * k_winding
            * num_slot
            * num_wire
            / (np.pi * bore_diameter)
        )

        partials[
            "surface_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_slot_number",
        ] = np.sqrt(2.0) * k_winding * num_wire * i_rms / (np.pi * bore_diameter)

        partials[
            "surface_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":wire_per_slot",
        ] = np.sqrt(2.0) * k_winding * num_slot * i_rms / (np.pi * bore_diameter)

        partials[
            "surface_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_factor",
        ] = np.sqrt(2.0) * num_slot * num_wire * i_rms / (np.pi * bore_diameter)

        partials[
            "surface_current_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = -(np.sqrt(2.0) * k_winding * num_slot * num_wire * i_rms / (np.pi * bore_diameter**2.0))

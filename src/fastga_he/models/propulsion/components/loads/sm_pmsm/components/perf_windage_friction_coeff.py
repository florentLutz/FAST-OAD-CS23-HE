# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesWindageFrictionCoefficient(om.ExplicitComponent):
    """
    Computation of the windage friction coefficients for use in mechanical losses
    estimation. The conditions are given by equation (II-73) and (II-76) in :cite:`touhami:2020`.
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

        self.add_input("airgap_reynolds_number", val=np.nan, shape=number_of_points)
        self.add_input("rotor_end_reynolds_number", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness",
            val=np.nan,
            units="m",
            desc="The distance between the rotor and the stator bore",
        )

        self.add_output("airgap_friction_coeff", val=np.zeros(number_of_points))
        self.add_output("rotor_end_friction_coeff", val=np.zeros(number_of_points))

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="airgap_friction_coeff",
            wrt="airgap_reynolds_number",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="rotor_end_friction_coeff",
            wrt="rotor_end_reynolds_number",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="airgap_friction_coeff",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        airgap_thickness = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness"
        ]
        re_airgap = inputs["airgap_reynolds_number"]
        re_rotor = inputs["rotor_end_reynolds_number"]

        conditions = [(re_airgap > 500.0) & (re_airgap < 1.0e4), re_airgap >= 1.0e4]

        choices = [
            0.515 * (airgap_thickness / rotor_radius) ** 0.3 * (re_airgap**-0.5),
            0.0325 * (airgap_thickness / rotor_radius) ** 0.3 * (re_airgap**-0.2),
        ]

        outputs["airgap_friction_coeff"] = np.select(
            conditions, choices, default=np.zeros_like(re_airgap)
        )
        outputs["rotor_end_friction_coeff"] = np.where(
            re_rotor < 3.0e5, 3.87 / re_rotor**0.5, 0.146 / re_rotor**0.2
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        rotor_radius = rotor_diameter / 2.0
        airgap_thickness = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness"
        ]
        re_airgap = inputs["airgap_reynolds_number"]
        re_rotor = inputs["rotor_end_reynolds_number"]

        conditions = [(re_airgap > 500.0) & (re_airgap < 1.0e4), re_airgap >= 1.0e4]

        d_cf_airgap_d_re_airgap = [
            -0.2575 * (airgap_thickness / rotor_radius) ** 0.3 * re_airgap**-1.5,
            -0.0065 * (airgap_thickness / rotor_radius) ** 0.3 * re_airgap**-1.2,
        ]

        d_cf_airgap_d_airgap_thickness = [
            0.1545 * airgap_thickness**-0.7 * rotor_radius**-0.3 * re_airgap**-0.5,
            0.00975 * airgap_thickness**-0.7 * rotor_radius**-0.3 * re_airgap**-0.2,
        ]

        d_cf_airgap_d_rotor_diameter = [
            -0.1545 * (2.0 * airgap_thickness) ** 0.3 * rotor_diameter**-1.3 * re_airgap**-0.5,
            -0.00975 * (2.0 * airgap_thickness) ** 0.3 * rotor_diameter**-1.3 * re_airgap**-0.2,
        ]

        partials["airgap_friction_coeff", "airgap_reynolds_number"] = np.select(
            conditions, d_cf_airgap_d_re_airgap, default=np.zeros_like(re_airgap)
        )

        partials[
            "airgap_friction_coeff",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness",
        ] = np.select(conditions, d_cf_airgap_d_airgap_thickness, default=np.zeros_like(re_airgap))

        partials[
            "airgap_friction_coeff",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = np.select(conditions, d_cf_airgap_d_rotor_diameter, default=np.zeros_like(re_airgap))

        partials["rotor_end_friction_coeff", "rotor_end_reynolds_number"] = np.where(
            re_rotor < 3.0e5, -1.935 / re_rotor**1.5, -0.0292 / re_rotor**1.2
        )

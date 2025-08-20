# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesWindageFrictionCoefficient(om.ExplicitComponent):
    """
    Computation of the friction coefficients resulting from windage in mechanical loss
    estimation.
    """

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("airgap_reynolds_number", val=np.nan, shape=number_of_points)
        self.add_input("rotor_end_reynolds_number", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness",
            val=np.nan,
            units="m",
        )

        self.add_output("airgap_friction_coeff", val=np.zeros(number_of_points))
        self.add_output("rotor_end_friction_coeff", val=np.zeros(number_of_points))

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
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
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        r_rot = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter"] / 2.0
        e_g = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness"]
        re_a = inputs["airgap_reynolds_number"]
        re_r = inputs["rotor_end_reynolds_number"]
        cf_a = np.zeros_like(re_a)

        # Conditions
        mask_1 = (re_a > 500) & (re_a < 1e4)
        mask_2 = re_a >= 1e4
        mask_3 = re_a <= 500

        cf_a[mask_1] = 0.515 * (e_g / r_rot) ** 0.3 * (re_a[mask_1] ** -0.5)
        cf_a[mask_2] = 0.0325 * (e_g / r_rot) ** 0.3 * (re_a[mask_2] ** -0.2)
        cf_a[mask_3] = 0.0

        outputs["airgap_friction_coeff"] = cf_a
        outputs["rotor_end_friction_coeff"] = np.where(
            re_r < 3e5, 3.87 / re_r**0.5, 0.146 / re_r**0.2
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        r_rot = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter"] / 2.0
        d_rot = r_rot * 2.0
        e_g = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness"]
        re_a = inputs["airgap_reynolds_number"]
        re_r = inputs["rotor_end_reynolds_number"]

        dcfdre = np.zeros_like(re_a)
        dcfde_g = np.zeros_like(re_a)
        dcfdrot = np.zeros_like(re_a)

        # Conditions
        mask_1 = (re_a > 500) & (re_a < 1e4)
        mask_2 = re_a >= 1e4
        mask_3 = re_a <= 500

        dcfdre[mask_1] = -0.2575 * (e_g / r_rot) ** 0.3 * (re_a[mask_1] ** -1.5)
        dcfdre[mask_2] = -0.0065 * (e_g / r_rot) ** 0.3 * (re_a[mask_2] ** -1.2)
        dcfdre[mask_3] = 0.0

        dcfde_g[mask_1] = 0.1545 * e_g**-0.7 * r_rot**-0.3 * (re_a[mask_1] ** -0.5)
        dcfde_g[mask_2] = 0.00975 * e_g**-0.7 * r_rot**-0.3 * (re_a[mask_2] ** -0.2)
        dcfde_g[mask_3] = 0.0

        dcfdrot[mask_1] = -0.1545 * (2.0 * e_g) ** 0.3 * d_rot**-1.3 * (re_a[mask_1] ** -0.5)
        dcfdrot[mask_2] = -0.00975 * (2.0 * e_g) ** 0.3 * d_rot**-1.3 * (re_a[mask_2] ** -0.2)
        dcfdrot[mask_3] = 0.0

        partials["airgap_friction_coeff", "airgap_reynolds_number"] = dcfdre

        partials[
            "airgap_friction_coeff",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Airgap_thickness",
        ] = dcfde_g

        partials[
            "airgap_friction_coeff",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rot_diameter",
        ] = dcfdrot

        partials["rotor_end_friction_coeff", "rotor_end_reynolds_number"] = np.where(
            re_r < 3e5, -1.935 / re_r**1.5, -0.0292 / re_r**1.2
        )

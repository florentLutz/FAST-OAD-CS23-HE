# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import POLAR_POINT_COUNT


class PerformancesBladesAero(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("elements_number", default=7, types=int)
        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "blades_aoa",
            val=np.full((elements_number, number_of_points), np.nan),
            units="deg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":alpha_list",
            val=np.full(POLAR_POINT_COUNT, np.nan),
            shape=POLAR_POINT_COUNT,
            units="deg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_array",
            val=np.full((elements_number, POLAR_POINT_COUNT), np.nan),
            shape=(elements_number, POLAR_POINT_COUNT),
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":cd_array",
            val=np.full((elements_number, POLAR_POINT_COUNT), np.nan),
            shape=(elements_number, POLAR_POINT_COUNT),
        )

        self.add_output(
            "blades_cl",
            val=np.full((elements_number, number_of_points), 0.5),
        )
        self.add_output(
            "blades_cd",
            val=np.full((elements_number, number_of_points), 0.01),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        alpha_list = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":alpha_list"
        ]
        cl_array = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_array"]
        cd_array = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":cd_array"]

        alpha = inputs["blades_aoa"]
        cl = np.zeros_like(alpha)
        cd = np.zeros_like(alpha)

        for idx, element_aoa in enumerate(alpha):

            alpha_element, cl_element, cd_element = self.reshape_polar(
                alpha_list,
                cl_array[idx, :],
                cd_array[idx, :],
            )
            cl[idx, :] = np.interp(element_aoa, alpha_element, cl_element)
            cd[idx, :] = np.interp(element_aoa, alpha_element, cd_element)

        outputs["blades_cl"] = cl
        outputs["blades_cd"] = cd

    @staticmethod
    def reshape_polar(alpha, c_l, c_d):
        """
        Reads the polar under the openmdao format (meaning with additional zeros and reshape
        so that only relevant angle are considered.

        Assumes that the AOA list is ordered.
        """
        idx_start = np.argmin(alpha)
        idx_end = np.argmax(alpha)

        return (
            alpha[idx_start : idx_end + 1],
            c_l[idx_start : idx_end + 1],
            c_d[idx_start : idx_end + 1],
        )

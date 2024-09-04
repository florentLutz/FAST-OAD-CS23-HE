# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

K0 = np.array([0.378269, 0.748135, -0.179986, -0.056464, -0.146746, -0.015255])
K1 = np.array([3.071020, -1.769885, 0.436595, 0.148643, -0.989332, 0.197940])
K2 = np.array([-2.827730, 2.054064, -0.467410, -0.277325, 0.698981, -0.008226])
K3 = np.array([0.997936, -0.916118, 0.199829, 0.157810, -0.143368, -0.057385])
K4 = np.array([-0.127645, 0.135543, -0.028919, -0.026546, 0.010470, 0.012221])

LIMIT_V_RATIO = (1.25, 2.25)
LIMIT_DISTANCE_RATIO = (0.25, 3.0)


class SlipstreamPropellerHeightImpactCoefficients(om.ExplicitComponent):
    """
    Implementation of the surrogate model from :cite:`patterson:2016` to account for height
    impact on the slipstream effect. This component will compute the values of the f_i
    coefficients, the computation of the beta will be done in a separate component.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]

        self.add_input("velocity_ratio_downstream", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
            val=np.nan,
            desc="Distance between the propeller and the wing leading edge as a ratio of the "
            "reference chord behind the propeller",
        )

        self.add_output(
            name="f_0",
            val=0.01,
            shape=number_of_points,
            desc="Order zero coefficient in the beta surrogate model",
        )
        self.add_output(
            name="f_1",
            val=0.01,
            shape=number_of_points,
            desc="First order coefficient in the beta surrogate model",
        )
        self.add_output(
            name="f_2",
            val=0.01,
            shape=number_of_points,
            desc="Second order coefficient in the beta surrogate model",
        )
        self.add_output(
            name="f_3",
            val=0.01,
            shape=number_of_points,
            desc="Third order coefficient in the beta surrogate model",
        )
        self.add_output(
            name="f_4",
            val=0.01,
            shape=number_of_points,
            desc="Fourth order coefficient in the beta surrogate model",
        )

        self.declare_partials(
            of="*",
            wrt="velocity_ratio_downstream",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        # Clip the input so that it matches the range used for the creation of the surrogate

        v_ratio = np.clip(inputs["velocity_ratio_downstream"], LIMIT_V_RATIO[0], LIMIT_V_RATIO[1])
        distance_ratio = np.clip(
            np.full_like(
                v_ratio,
                inputs[
                    "data:propulsion:he_power_train:propeller:"
                    + propeller_id
                    + ":from_wing_LE_ratio"
                ],
            ),
            LIMIT_DISTANCE_RATIO[0],
            LIMIT_DISTANCE_RATIO[1],
        )

        x_matrix = np.empty(shape=(6, number_of_points))
        x_matrix[0, :] = np.ones(number_of_points)
        x_matrix[1, :] = distance_ratio
        x_matrix[2, :] = distance_ratio**2.0
        x_matrix[3, :] = distance_ratio * v_ratio
        x_matrix[4, :] = v_ratio
        x_matrix[5, :] = v_ratio**2.0

        outputs["f_0"] = np.matmul(K0, x_matrix)
        outputs["f_1"] = np.matmul(K1, x_matrix)
        outputs["f_2"] = np.matmul(K2, x_matrix)
        outputs["f_3"] = np.matmul(K3, x_matrix)
        outputs["f_4"] = np.matmul(K4, x_matrix)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        v_ratio_unclipped = inputs["velocity_ratio_downstream"]
        v_ratio = np.clip(v_ratio_unclipped, LIMIT_V_RATIO[0], LIMIT_V_RATIO[1])

        distance_ratio_unclipped = np.full_like(
            v_ratio,
            inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio"
            ],
        )
        distance_ratio = np.clip(
            distance_ratio_unclipped, LIMIT_DISTANCE_RATIO[0], LIMIT_DISTANCE_RATIO[1]
        )

        d_x_matrix_d_v = np.empty(shape=(6, number_of_points))
        d_x_matrix_d_v[0, :] = np.zeros(number_of_points)
        d_x_matrix_d_v[1, :] = np.zeros(number_of_points)
        d_x_matrix_d_v[2, :] = np.zeros(number_of_points)
        d_x_matrix_d_v[3, :] = distance_ratio
        d_x_matrix_d_v[4, :] = np.ones(number_of_points)
        d_x_matrix_d_v[5, :] = 2.0 * v_ratio

        d_x_matrix_d_d = np.empty(shape=(6, number_of_points))
        d_x_matrix_d_d[0, :] = np.zeros(number_of_points)
        d_x_matrix_d_d[1, :] = np.ones(number_of_points)
        d_x_matrix_d_d[2, :] = 2.0 * distance_ratio
        d_x_matrix_d_d[3, :] = v_ratio
        d_x_matrix_d_d[4, :] = np.zeros(number_of_points)
        d_x_matrix_d_d[5, :] = np.zeros(number_of_points)

        partials["f_0", "velocity_ratio_downstream"] = np.where(
            v_ratio == v_ratio_unclipped, np.matmul(K0, d_x_matrix_d_v), 0
        )
        partials[
            "f_0",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
        ] = np.where(distance_ratio == distance_ratio_unclipped, np.matmul(K0, d_x_matrix_d_d), 0)

        partials["f_1", "velocity_ratio_downstream"] = np.where(
            v_ratio == v_ratio_unclipped, np.matmul(K1, d_x_matrix_d_v), 0
        )
        partials[
            "f_1",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
        ] = np.where(distance_ratio == distance_ratio_unclipped, np.matmul(K1, d_x_matrix_d_d), 0)

        partials["f_2", "velocity_ratio_downstream"] = np.where(
            v_ratio == v_ratio_unclipped, np.matmul(K2, d_x_matrix_d_v), 0
        )
        partials[
            "f_2",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
        ] = np.where(distance_ratio == distance_ratio_unclipped, np.matmul(K2, d_x_matrix_d_d), 0)

        partials["f_3", "velocity_ratio_downstream"] = np.where(
            v_ratio == v_ratio_unclipped, np.matmul(K3, d_x_matrix_d_v), 0
        )
        partials[
            "f_3",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
        ] = np.where(distance_ratio == distance_ratio_unclipped, np.matmul(K3, d_x_matrix_d_d), 0)

        partials["f_4", "velocity_ratio_downstream"] = np.where(
            v_ratio == v_ratio_unclipped, np.matmul(K4, d_x_matrix_d_v), 0
        )
        partials[
            "f_4",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
        ] = np.where(distance_ratio == distance_ratio_unclipped, np.matmul(K4, d_x_matrix_d_d), 0)

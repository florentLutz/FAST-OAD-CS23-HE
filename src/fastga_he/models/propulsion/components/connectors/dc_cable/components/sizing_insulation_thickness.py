# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingInsulationThickness(om.ExplicitComponent):
    """
    Computation of insulation thickness.

    Based on the formula from :cite:`aretskin:2021`
    """

    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":conductor:radius",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":voltage_caliber",
            val=np.nan,
            units="V",
        )

        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:dielectric_permittivity",
            val=4,
            desc="Dielectric permittivity of the insulation, chosen as Gexol insulation",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:void_thickness",
            units="m",
            val=0.005e-2,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:breakdown_voltage",
            units="V",
            val=340,
            desc="Minimum breakdown voltage of air cavity",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:k_factor_radius",
            val=1.0,
            desc="K-factor to increase the radius of the insulation",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness",
            units="m",
            val=1e-3,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        radius_conductor = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius"
        ]
        dielectric_permittivity = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:dielectric_permittivity"
        ]
        v_max = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber"
        ]
        t_v = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:void_thickness"
        ]
        alpha = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:breakdown_voltage"
        ]
        k_factor = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:k_factor_radius"
        ]

        shape_factor = 3.0 * dielectric_permittivity / (1.0 + 2.0 * dielectric_permittivity)

        if v_max > 20000.0:
            c_factor = 0.0
        else:
            c_factor = 0.1e-2

        if v_max >= 10000.0:
            v_max *= np.sqrt(2.0 / 3.0)

        factor_exp = (shape_factor * v_max * t_v) / (alpha * radius_conductor)

        t_i = radius_conductor * (np.exp(factor_exp) - 1.0) + c_factor

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness"
        ] = t_i * k_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        output_str = (
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:thickness"
        )

        radius_conductor = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius"
        ]
        dielectric_permittivity = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:dielectric_permittivity"
        ]
        v_max = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber"
        ]
        t_v = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:void_thickness"
        ]
        alpha = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:breakdown_voltage"
        ]
        k_factor = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:k_factor_radius"
        ]

        shape_factor = 3.0 * dielectric_permittivity / (1.0 + 2.0 * dielectric_permittivity)
        d_sf_d_permittivity = 3.0 / (1.0 + 2.0 * dielectric_permittivity) ** 2.0

        if v_max > 10000:
            d_factor_d_v_max = np.sqrt(2.0 / 3.0) * shape_factor * t_v / (alpha * radius_conductor)
            v_max *= np.sqrt(2.0 / 3.0)
        else:
            d_factor_d_v_max = shape_factor * t_v / (alpha * radius_conductor)

        if v_max > 20000.0:
            c_factor = 0.0
        else:
            c_factor = 0.1e-2

        factor_exp = (shape_factor * v_max * t_v) / (alpha * radius_conductor)
        d_factor_d_sf = v_max * t_v / (alpha * radius_conductor)
        d_factor_d_t_v = shape_factor * v_max / (alpha * radius_conductor)
        d_factor_d_alpha = -(shape_factor * v_max * t_v) / (alpha**2.0 * radius_conductor)
        d_factor_d_r_c = -(shape_factor * v_max * t_v) / (alpha * radius_conductor**2.0)

        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":conductor:radius",
        ] = (np.exp(factor_exp) - 1.0) + radius_conductor * np.exp(
            factor_exp
        ) * d_factor_d_r_c * k_factor
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_caliber",
        ] = radius_conductor * np.exp(factor_exp) * d_factor_d_v_max * k_factor
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:dielectric_permittivity",
        ] = radius_conductor * np.exp(factor_exp) * d_factor_d_sf * d_sf_d_permittivity * k_factor
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:void_thickness",
        ] = radius_conductor * np.exp(factor_exp) * d_factor_d_t_v * k_factor
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:breakdown_voltage",
        ] = radius_conductor * np.exp(factor_exp) * d_factor_d_alpha * k_factor
        partials[
            output_str,
            "settings:propulsion:he_power_train:DC_cable_harness:insulation:breakdown_voltage",
        ] = radius_conductor * np.exp(factor_exp) * d_factor_d_alpha * k_factor
        partials[
            output_str,
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":insulation:k_factor_radius",
        ] = radius_conductor * (np.exp(factor_exp) - 1.0) + c_factor

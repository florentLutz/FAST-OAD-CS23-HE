# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingBusBarInsulationThickness(om.ExplicitComponent):
    """
    The thickness that separates the two planes of the bus bar depends on the partial discharge
    criteria. We will reuse the model we used for sizing the thickness of cables and
    adapt the inputs.

    Based on the formula from :cite:`aretskin:2021`
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
            units="V",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity",
            val=4,
            desc="Dielectric permittivity of the insulation, chosen as Gexol insulation",
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_bus:insulation:void_thickness",
            units="m",
            val=0.005e-2,
        )
        self.add_input(
            "settings:propulsion:he_power_train:DC_bus:insulation:breakdown_voltage",
            units="V",
            val=340,
            desc="Mininum breakdown voltage of air cavity",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            units="m",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        thickness_conductor = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness"
        ]
        dielectric_permittivity = inputs[
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity"
        ]
        v_max = inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber"]
        t_v = inputs["settings:propulsion:he_power_train:DC_bus:insulation:void_thickness"]
        alpha = inputs["settings:propulsion:he_power_train:DC_bus:insulation:breakdown_voltage"]

        shape_factor = 3.0 * dielectric_permittivity / (1.0 + 2.0 * dielectric_permittivity)

        if v_max > 20000.0:
            c_factor = 0.0
        else:
            c_factor = 0.1e-2

        if v_max >= 10000.0:
            v_max *= np.sqrt(2.0 / 3.0)

        factor_exp = (shape_factor * v_max * t_v) / (alpha * thickness_conductor)

        t_i = thickness_conductor * (np.exp(factor_exp) - 1.0) + c_factor

        outputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness"
        ] = t_i

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        thickness_conductor = inputs[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness"
        ]
        dielectric_permittivity = inputs[
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity"
        ]
        v_max = inputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber"]
        t_v = inputs["settings:propulsion:he_power_train:DC_bus:insulation:void_thickness"]
        alpha = inputs["settings:propulsion:he_power_train:DC_bus:insulation:breakdown_voltage"]

        shape_factor = 3.0 * dielectric_permittivity / (1.0 + 2.0 * dielectric_permittivity)
        d_sf_d_permittivity = 3.0 / (1.0 + 2.0 * dielectric_permittivity) ** 2.0

        if v_max > 10000:
            d_factor_d_v_max = (
                np.sqrt(2.0 / 3.0) * shape_factor * t_v / (alpha * thickness_conductor)
            )
            v_max *= np.sqrt(2.0 / 3.0)
        else:
            d_factor_d_v_max = shape_factor * t_v / (alpha * thickness_conductor)

        factor_exp = (shape_factor * v_max * t_v) / (alpha * thickness_conductor)
        d_factor_d_sf = v_max * t_v / (alpha * thickness_conductor)
        d_factor_d_t_v = shape_factor * v_max / (alpha * thickness_conductor)
        d_factor_d_alpha = -(shape_factor * v_max * t_v) / (alpha ** 2.0 * thickness_conductor)
        d_factor_d_r_c = -(shape_factor * v_max * t_v) / (alpha * thickness_conductor ** 2.0)

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":cross_section:thickness",
        ] = (np.exp(factor_exp) - 1.0) + thickness_conductor * np.exp(factor_exp) * d_factor_d_r_c
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_caliber",
        ] = (
            thickness_conductor * np.exp(factor_exp) * d_factor_d_v_max
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            "settings:propulsion:he_power_train:DC_bus:insulation:dielectric_permittivity",
        ] = (
            thickness_conductor * np.exp(factor_exp) * d_factor_d_sf * d_sf_d_permittivity
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            "settings:propulsion:he_power_train:DC_bus:insulation:void_thickness",
        ] = (
            thickness_conductor * np.exp(factor_exp) * d_factor_d_t_v
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            "settings:propulsion:he_power_train:DC_bus:insulation:breakdown_voltage",
        ] = (
            thickness_conductor * np.exp(factor_exp) * d_factor_d_alpha
        )
        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":insulation:thickness",
            "settings:propulsion:he_power_train:DC_bus:insulation:breakdown_voltage",
        ] = (
            thickness_conductor * np.exp(factor_exp) * d_factor_d_alpha
        )

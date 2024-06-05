# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import (
    SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE,
    SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE,
)
from .perf_temperature import SUBMODEL_DC_LINE_TEMPERATURE_STEADY_STATE

SUBMODEL_DC_LINE_RESISTANCE_NO_LOOP = (
    "fastga_he.submodel.propulsion.performances.dc_line.resistance_profile.no_loop"
)

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(
    SUBMODEL_DC_LINE_PERFORMANCES_RESISTANCE_PROFILE, SUBMODEL_DC_LINE_RESISTANCE_NO_LOOP
)
class PerformancesResistanceNoLoop(om.ExplicitComponent):
    """
    This variation of the resistance module has been created after the realization that if the
    thermal dynamics of the cable is ignored (steady state), an explicit expression of the
    resistance could be obtained by reformulating the formulas which would suppress the need to
    iterate on the temperature while still providing it. This means that if this model is to be
    used, it HAS to be with the steady-state temperature model, hence why a warning is raised if
    the proper submodel is not used.

    In those condition, the resistance is the solution of the following polynomial expression:

    R**2 - R * R_ref * (1 + alpha * (T_ext - T_ref)) - alpha * R_ref * Delta_U**2 * n_c**2 / h / S
    = 0
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        if (
            oad.RegisterSubmodel.active_models[SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE]
            != SUBMODEL_DC_LINE_TEMPERATURE_STEADY_STATE
        ):
            _LOGGER.warning(
                "Submodel fastga_he.submodel.propulsion.performances.dc_line.resistance_profile."
                "no_loop is meant to be used in conjunction with submodel "
                + SUBMODEL_DC_LINE_TEMPERATURE_STEADY_STATE
                + " for service "
                + SUBMODEL_DC_LINE_PERFORMANCES_TEMPERATURE_PROFILE
                + ". Code will converge but results on cable performances will be wrong."
            )

        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
            val=np.nan,
            units="degK**-1",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            name="settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature",
            val=293.15,
            units="degK",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1.0,
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the expected output (current flows from source/input to load/output)",
            shape=number_of_points,
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the expected input (current flows from source/input to load/output)",
            shape=number_of_points,
        )
        self.add_input(
            "heat_transfer_coefficient",
            val=np.full(number_of_points, 50.0),
            units="W/m**2/degK",
            desc="Heat transfer coefficient between cable and outside medium",
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature outside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "resistance_per_cable",
            val=np.full(number_of_points, 1.0e-3),
            units="ohm",
            shape=number_of_points,
        )

        self.declare_partials(
            of="resistance_per_cable",
            wrt=[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
                "settings:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:reference_temperature",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":cable:resistance",
                "data:propulsion:he_power_train:DC_cable_harness:"
                + harness_id
                + ":properties:resistance_temperature_scale_factor",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="resistance_per_cable",
            wrt=[
                "dc_voltage_out",
                "dc_voltage_in",
                "exterior_temperature",
                "heat_transfer_coefficient",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        alpha_r = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor"
        ]
        reference_resistance = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance"
        ]
        reference_temperature = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature"
        ]
        number_cables = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
        ]
        voltage_out = inputs["dc_voltage_out"]
        voltage_in = inputs["dc_voltage_in"]
        cable_radius = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ]
        cable_length = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"
        ]

        h = inputs["heat_transfer_coefficient"]

        temp_ext = inputs["exterior_temperature"]

        b_term = reference_resistance * (1.0 + alpha_r * (temp_ext - reference_temperature))
        c_term = (
            alpha_r
            * reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )

        resistance = (b_term + np.sqrt(b_term ** 2.0 + 4.0 * c_term)) / 2.0

        outputs["resistance_per_cable"] = resistance

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        alpha_r = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor"
        ]
        reference_resistance = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance"
        ]
        reference_temperature = inputs[
            "settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature"
        ]
        number_cables = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
        ]
        voltage_out = inputs["dc_voltage_out"]
        voltage_in = inputs["dc_voltage_in"]
        cable_radius = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius"
        ]
        cable_length = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length"
        ]

        h = inputs["heat_transfer_coefficient"]

        temp_ext = inputs["exterior_temperature"]

        b_term = reference_resistance * (1.0 + alpha_r * (temp_ext - reference_temperature))
        c_term = (
            alpha_r
            * reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )

        d_resistance_d_b_term = 0.5 * (1.0 + b_term / np.sqrt(b_term ** 2 + 4.0 * c_term))
        d_resistance_d_c_term = 1.0 / np.sqrt(b_term ** 2 + 4.0 * c_term)

        d_b_term_d_alpha_r = reference_resistance * (temp_ext - reference_temperature)
        d_c_term_d_alpha_r = (
            reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":properties:resistance_temperature_scale_factor",
        ] = (
            d_resistance_d_b_term * d_b_term_d_alpha_r + d_resistance_d_c_term * d_c_term_d_alpha_r
        )

        d_b_term_d_r_ref = 1.0 + alpha_r * (temp_ext - reference_temperature)
        d_c_term_d_r_ref = (
            alpha_r
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:resistance",
        ] = (
            d_resistance_d_b_term * d_b_term_d_r_ref + d_resistance_d_c_term * d_c_term_d_r_ref
        )

        d_b_term_d_t_ext = reference_resistance * alpha_r
        d_c_term_d_t_ext = 0.0
        partials["resistance_per_cable", "exterior_temperature"] = (
            d_resistance_d_b_term * d_b_term_d_t_ext + d_resistance_d_c_term * d_c_term_d_t_ext
        )

        d_b_term_d_t_ref = -reference_resistance * alpha_r
        d_c_term_d_t_ref = 0.0
        partials[
            "resistance_per_cable",
            "settings:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":cable:reference_temperature",
        ] = (
            d_resistance_d_b_term * d_b_term_d_t_ref + d_resistance_d_c_term * d_c_term_d_t_ref
        )

        d_b_term_d_nc = 0.0
        d_c_term_d_nc = (
            alpha_r
            * reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * 2.0
            * number_cables
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = (
            d_resistance_d_b_term * d_b_term_d_nc + d_resistance_d_c_term * d_c_term_d_nc
        )

        d_b_term_d_v_out = 0.0
        d_c_term_d_v_out = -(
            alpha_r
            * reference_resistance
            * 2.0
            * (voltage_in - voltage_out)
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )
        partials["resistance_per_cable", "dc_voltage_out"] = (
            d_resistance_d_b_term * d_b_term_d_v_out + d_resistance_d_c_term * d_c_term_d_v_out
        )

        d_b_term_d_v_in = 0.0
        d_c_term_d_v_in = (
            alpha_r
            * reference_resistance
            * 2.0
            * (voltage_in - voltage_out)
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length)
        )
        partials["resistance_per_cable", "dc_voltage_in"] = (
            d_resistance_d_b_term * d_b_term_d_v_in + d_resistance_d_c_term * d_c_term_d_v_in
        )

        d_b_term_d_h = 0.0
        d_c_term_d_h = -(
            alpha_r
            * reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h ** 2.0
            / (2.0 * np.pi * cable_radius * cable_length)
        )
        partials["resistance_per_cable", "heat_transfer_coefficient"] = (
            d_resistance_d_b_term * d_b_term_d_h + d_resistance_d_c_term * d_c_term_d_h
        )

        d_b_term_d_r = 0.0
        d_c_term_d_r = -(
            alpha_r
            * reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius ** 2.0 * cable_length)
        )
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":cable:radius",
        ] = (
            d_resistance_d_b_term * d_b_term_d_r + d_resistance_d_c_term * d_c_term_d_r
        )

        d_b_term_d_l = 0.0
        d_c_term_d_l = -(
            alpha_r
            * reference_resistance
            * (voltage_in - voltage_out) ** 2.0
            * number_cables ** 2.0
            / h
            / (2.0 * np.pi * cable_radius * cable_length ** 2.0)
        )
        partials[
            "resistance_per_cable",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":length",
        ] = (
            d_resistance_d_b_term * d_b_term_d_l + d_resistance_d_c_term * d_c_term_d_l
        )

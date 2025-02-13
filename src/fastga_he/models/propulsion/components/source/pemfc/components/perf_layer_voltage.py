# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from ..constants import SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE
import fastoad.api as oad

DEFAULT_LAYER_VOLTAGE = 0.7
DEFAULT_PRESSURE_ATM = 1.0
DEFAULT_TEMPERATURE = 288.15

oad.RegisterSubmodel.active_models[SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE] = (
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.simple"
)


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE,
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.simple",
)
class PerformancesSinglePEMFCVoltageSimple(om.ExplicitComponent):
    # TODO: Edit citation after rebase
    """
    Computation of the voltage of single layer proton exchange membrane fuel cell inside one
    stack. Assumes it can be estimated with the i-v curve relation. Model based on existing
    pemfc, Aerostack Ultralight 200, details can be found in :cite:`Fuel Cell and Battery Hybrid
    System Optimization by J. Hoogendoorn:2018`.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            "open_circuit_voltage",
            default=0.83,
            desc="open_circuit_voltage of one layer of pemfc [V]",
        )

        self.options.declare(
            "activation_loss_coefficient",
            default=0.014,
            desc="activation loss coefficient of one layer of pemfc (V/ln(A/cm**2))",
        )

        self.options.declare(
            "ohmic_resistance",
            default=0.24,
            desc="ohmic resistance of one layer of pemfc [V/ln(A/cm**2)]",
        )

        self.options.declare(
            "coefficient_in_concentration_loss",
            default=5.63 * 10**-6,
            desc="coefficient in concentration loss of one layer of pemfc [V]",
        )

        self.options.declare(
            "exponential_coefficient_in_concentration_loss",
            default=11.42,
            desc="exponential coefficient in concentration loss of one layer of pemfc [cm**2/A]",
        )

        self.options.declare(
            "exponential_coefficient_in_concentration_loss",
            default=11.42,
            desc="exponential coefficient in concentration loss of one layer of pemfc [cm**2/A]",
        )

        self.options.declare(
            "max_current_density", default=0.7, desc="maximum current density  of pemfc[A/cm^2]"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        self.add_input(
            "fc_current_density",
            units="A/cm**2",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            name="nominal_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
            desc="The nominal pressure at which the PEMFC operates does not affect the layer "
            "voltage ",
        )

        self.add_input(
            "operation_pressure",
            units="atm",
            val=np.full(number_of_points, DEFAULT_PRESSURE_ATM),
        )

        self.add_output(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.full(number_of_points, DEFAULT_LAYER_VOLTAGE),
        )

        self.add_output(
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
        )

        self.declare_partials(
            of="single_layer_pemfc_voltage",
            wrt=["fc_current_density", "operation_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="single_layer_pemfc_voltage",
            wrt="nominal_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
            wrt="nominal_pressure",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        voc = self.options["open_circuit_voltage"]
        active_loss_coeff = self.options["activation_loss_coefficient"]
        r = self.options["ohmic_resistance"]
        m = self.options["coefficient_in_concentration_loss"]
        n = self.options["exponential_coefficient_in_concentration_loss"]

        i = np.clip(
            inputs["fc_current_density"],
            np.full_like(inputs["fc_current_density"], 1e-2),
            np.full_like(inputs["fc_current_density"], self.options["max_current_density"]),
        )

        operation_pressure = inputs["operation_pressure"]

        nominal_pressure = inputs["nominal_pressure"]

        pressure_ratio_log = np.log(operation_pressure / nominal_pressure)

        pressure_coeff = -0.0032 * pressure_ratio_log**2 + 0.0019 * pressure_ratio_log + 0.0542

        outputs["single_layer_pemfc_voltage"] = (
            voc
            - active_loss_coeff * np.log(i)
            - r * i
            - m * np.exp(n * i)
            + pressure_coeff * pressure_ratio_log
        )
        # This output is to for tank connection
        outputs[
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ] = nominal_pressure

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        active_loss_coeff = self.options["activation_loss_coefficient"]
        r = self.options["ohmic_resistance"]
        m = self.options["coefficient_in_concentration_loss"]
        n = self.options["exponential_coefficient_in_concentration_loss"]

        operation_pressure = inputs["operation_pressure"]

        nominal_pressure = inputs["nominal_pressure"]

        pressure_ratio_log = np.log(operation_pressure / nominal_pressure)

        i = np.clip(
            inputs["fc_current_density"],
            np.full_like(inputs["fc_current_density"], 1e-2),
            np.full_like(inputs["fc_current_density"], self.options["max_current_density"]),
        )

        partials_j = np.where(
            inputs["fc_current_density"] == i,
            -active_loss_coeff / i - r - m * n * np.exp(n * i),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials["single_layer_pemfc_voltage", "operation_pressure"] = -(
            48 * pressure_ratio_log**2 - 19 * pressure_ratio_log - 271
        ) / (5000 * operation_pressure)

        partials["single_layer_pemfc_voltage", "nominal_pressure"] = (
            48 * pressure_ratio_log**2 - 19 * pressure_ratio_log - 271
        ) / (5000 * nominal_pressure)


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE,
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.analytical",
)
class PerformancesSinglePEMFCVoltageAnalytical(om.ExplicitComponent):
    """
    Computation of the voltage of single layer proton exchange membrane fuel cell inside one
    stack. Assumes it can be estimated with the i-v curve relation. Model based on analytical i-v
    curve equation, details can be found in: cite:`Preliminary Propulsion System Sizing Methods
    for PEM Fuel Cell Aircraft by D.Juschus:2021`.
    """

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            "reversible_electric_potential",
            default=1.229,
            desc="reversible electric potential of one layer of pemfc [V]",
        )

        self.options.declare(
            "gas_constant",
            default=8.314,
            desc="gas constant in [J/(mol*K)]",
        )

        self.options.declare(
            "faraday_constant",
            default=96485.3321,
            desc="faraday constant in [sec*A/mol]",
        )

        self.options.declare(
            "entropy_difference",
            default=44.34,
            desc="entropy difference from reation in [J/(mol*K)]",
        )

        self.options.declare(
            "standard_temperature",
            default=289.15,
            desc="standard temperature in [K]",
        )

        self.options.declare(
            "cathode_transfer_coefficient",
            default=0.3,
            desc="transfer_coefficient at the cathode side of fuel cell",
        )

        self.options.declare(
            "mass_transport_loss_constant",
            default=0.5,
            desc="the constant result from mass transport in pemfc [V]",
        )

        self.options.declare(
            "area_specific_resistance",
            default=1.0 * 10**-6,
            desc="Combined ohmic resistance that leads to losses in pemfc [Ωm**2]",
        )

        self.options.declare(
            "limiting_current_density",
            default=20000.0,
            desc="low limit for current density of pemfc [A/m**2]",
        )

        self.options.declare(
            "leakage_current_density",
            default=100.0,
            desc="leak loss of  current density from pemfc [A/m**2]",
        )

        self.options.declare(
            "max_current_density",
            default=7e3,
            desc="maximum current density  of pemfc",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        self.add_input(
            "fc_current_density",
            units="A/m**2",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            name="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
        )

        self.add_input(
            "operation_pressure",
            units="atm",
            val=np.full(number_of_points, DEFAULT_PRESSURE_ATM),
        )

        self.add_input(
            "operation_temperature",
            units="K",
            val=np.full(number_of_points, DEFAULT_TEMPERATURE),
        )

        self.add_input(
            name="analytical_voltage_adjust_factor",
            val=np.full(number_of_points, 1.0),
        )

        self.add_output(
            "single_layer_pemfc_voltage",
            units="V",
            val=np.full(number_of_points, DEFAULT_LAYER_VOLTAGE),
        )

        self.declare_partials(
            of="*",
            wrt=[
                "fc_current_density",
                "operation_pressure",
                "operation_temperature",
                "analytical_voltage_adjust_factor",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        e0 = self.options["reversible_electric_potential"]
        ds = self.options["entropy_difference"]
        gas_const = self.options["gas_constant"]
        faraday_const = self.options["faraday_constant"]
        t0 = self.options["standard_temperature"] * np.ones(number_of_points)
        a = self.options["cathode_transfer_coefficient"]
        r = self.options["area_specific_resistance"]
        c = self.options["mass_transport_loss_constant"]
        jlim = self.options["limiting_current_density"] * np.ones(number_of_points)
        jleak = self.options["leakage_current_density"] * np.ones(number_of_points)

        vf = inputs["analytical_voltage_adjust_factor"]

        j = np.clip(
            inputs["fc_current_density"],
            np.full_like(inputs["fc_current_density"], 10.0),
            np.full_like(inputs["fc_current_density"], self.options["max_current_density"]),
        )

        p_o2 = inputs["operation_pressure"]

        p_h2 = inputs[
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ]

        t = inputs["operation_temperature"]

        outputs["single_layer_pemfc_voltage"] = vf * (
            e0
            - ds / (2 * faraday_const) * (t - t0)
            + gas_const * t / (2 * faraday_const) * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - gas_const * t / (2 * a * faraday_const) * np.log(j + jleak)
            - r * j
            - c * np.log(jlim / (jlim - j - jleak))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        e0 = self.options["reversible_electric_potential"]
        ds = self.options["entropy_difference"]
        gas_const = self.options["gas_constant"]
        faraday_const = self.options["faraday_constant"]
        t0 = self.options["standard_temperature"] * np.ones(number_of_points)
        t = inputs["operation_temperature"]
        a = self.options["cathode_transfer_coefficient"]
        r = self.options["area_specific_resistance"]
        c = self.options["mass_transport_loss_constant"]
        jlim = self.options["limiting_current_density"] * np.ones(number_of_points)
        jleak = self.options["leakage_current_density"] * np.ones(number_of_points)
        p_o2 = inputs["operation_pressure"]
        vf = inputs["analytical_voltage_adjust_factor"]

        p_h2 = inputs[
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ]

        j = np.clip(
            inputs["fc_current_density"],
            np.full_like(inputs["fc_current_density"], 10.0),
            np.full_like(inputs["fc_current_density"], self.options["max_current_density"]),
        )

        partials_j = np.where(
            inputs["fc_current_density"] == j,
            vf
            * (
                -gas_const * t / (2 * faraday_const * a * (j + jleak))
                - c / (-j + jlim - jleak)
                - r * np.ones(number_of_points)
            ),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials["single_layer_pemfc_voltage", "operation_temperature"] = vf * (
            -ds / (2 * faraday_const) * np.ones(number_of_points)
            + gas_const / (2 * faraday_const) * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - gas_const / (2 * a * faraday_const) * np.log(j + jleak)
        )

        partials["single_layer_pemfc_voltage", "operation_pressure"] = vf * (
            gas_const * t / (4 * faraday_const * p_o2)
        )

        partials[
            "single_layer_pemfc_voltage",
            "data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
        ] = vf * (gas_const * t / (2 * faraday_const * p_h2))

        partials["single_layer_pemfc_voltage", "analytical_voltage_adjust_factor"] = (
            e0
            - ds / (2 * faraday_const) * (t - t0)
            + gas_const * t / (2 * faraday_const) * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - gas_const * t / (2 * a * faraday_const) * np.log(j + jleak)
            - r * j
            - c * np.log(jlim / (jlim - j - jleak))
        )

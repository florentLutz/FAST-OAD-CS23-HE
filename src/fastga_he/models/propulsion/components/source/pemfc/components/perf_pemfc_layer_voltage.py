# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE

DEFAULT_LAYER_VOLTAGE = 0.7  # [V]
DEFAULT_PRESSURE_ATM = 1.0  # [atm]
DEFAULT_TEMPERATURE = 288.15  # [K]

oad.RegisterSubmodel.active_models[SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE] = (
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.simple"
)


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE,
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.simple",
)
class PerformancesPEMFCStackSingleVoltageSimple(om.ExplicitComponent):
    """
    Computation of the single layer voltage of PEMFC. Model based on existing PEMFC, Aerostack
    Ultralight 200W, details can be found in :cite:`hoogendoorn:2018`.
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
            desc="activation loss coefficient of one layer of pemfc [V/ln(A/cm**2)]",
        )

        self.options.declare(
            "ohmic_resistance",
            default=0.24,
            desc="ohmic resistance of one layer of pemfc [ohm*cm**2)]",
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
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
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
            "operating_pressure",
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
            wrt=["fc_current_density", "operating_pressure"],
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
        ocv = self.options["open_circuit_voltage"]
        active_loss_coeff = self.options["activation_loss_coefficient"]
        resistance = self.options["ohmic_resistance"]
        m_loss = self.options["coefficient_in_concentration_loss"]
        n_loss = self.options["exponential_coefficient_in_concentration_loss"]

        i_clipped = np.clip(inputs["fc_current_density"], 1e-2, self.options["max_current_density"])

        operating_pressure = inputs["operating_pressure"]

        nominal_pressure = inputs["nominal_pressure"]

        pressure_ratio_log = np.log(operating_pressure / nominal_pressure)

        pressure_coeff = -0.0032 * pressure_ratio_log**2 + 0.0019 * pressure_ratio_log + 0.0542

        outputs["single_layer_pemfc_voltage"] = (
            ocv
            - active_loss_coeff * np.log(i_clipped)
            - resistance * i_clipped
            - m_loss * np.exp(n_loss * i_clipped)
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
        resistance = self.options["ohmic_resistance"]
        m_loss = self.options["coefficient_in_concentration_loss"]
        n_loss = self.options["exponential_coefficient_in_concentration_loss"]

        operating_pressure = inputs["operating_pressure"]

        nominal_pressure = inputs["nominal_pressure"]

        pressure_ratio_log = np.log(operating_pressure / nominal_pressure)

        i_clipped = np.clip(inputs["fc_current_density"], 1e-2, self.options["max_current_density"])

        partials_j = np.where(
            inputs["fc_current_density"] == i_clipped,
            -active_loss_coeff / i_clipped
            - resistance
            - m_loss * n_loss * np.exp(n_loss * i_clipped),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials["single_layer_pemfc_voltage", "operating_pressure"] = -(
            48 * pressure_ratio_log**2 - 19 * pressure_ratio_log - 271
        ) / (5000 * operating_pressure)

        partials["single_layer_pemfc_voltage", "nominal_pressure"] = (
            48 * pressure_ratio_log**2 - 19 * pressure_ratio_log - 271
        ) / (5000 * nominal_pressure)


@oad.RegisterSubmodel(
    SUBMODEL_PERFORMANCES_PEMFC_LAYER_VOLTAGE,
    "fastga_he.submodel.propulsion.performances.pemfc.layer_voltage.analytical",
)
class PerformancesPEMFCStackSingleVoltageAnalytical(om.ExplicitComponent):
    """
    Computation of the single layer voltage of PEMFC. Model based on analytical i-v curve equation,
    details can be found in :cite:`Juschus:2021`.
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
            desc="Combined ohmic resistance that leads to losses in pemfc [ohm*m**2]",
        )

        self.options.declare(
            "leakage_current_density",
            default=100.0,
            desc="leak loss of  current density from pemfc [A/m**2]",
        )

        self.options.declare(
            "max_current_density",
            default=0.7,
            desc="maximum current density of pemfc [A/cm**2]",
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
            name="data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
        )

        self.add_input(
            "operating_pressure",
            units="atm",
            val=np.full(number_of_points, DEFAULT_PRESSURE_ATM),
        )

        self.add_input(
            "operating_temperature",
            units="K",
            val=np.full(number_of_points, DEFAULT_TEMPERATURE),
        )

        self.add_input(
            name="ambient_pressure_voltage_correction",
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
                "operating_pressure",
                "operating_temperature",
                "ambient_pressure_voltage_correction",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack:"
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
        t0 = np.full(number_of_points, self.options["standard_temperature"])
        a_transfer = self.options["cathode_transfer_coefficient"]
        resistance = self.options["area_specific_resistance"]
        c_loss = self.options["mass_transport_loss_constant"]
        jlim = np.full(number_of_points, self.options["max_current_density"] * 10000)
        jleak = np.full(number_of_points, self.options["leakage_current_density"])

        pvc = inputs["ambient_pressure_voltage_correction"]

        j_clipped = np.clip(
            inputs["fc_current_density"], 10.0, self.options["max_current_density"] * 10000
        )

        p_o2 = inputs["operating_pressure"]

        p_h2 = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ]

        temp_op = inputs["operating_temperature"]

        outputs["single_layer_pemfc_voltage"] = pvc * (
            e0
            - ds / (2 * faraday_const) * (temp_op - t0)
            + gas_const * temp_op / (2 * faraday_const) * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - gas_const * temp_op / (2 * a_transfer * faraday_const) * np.log(j_clipped + jleak)
            - resistance * j_clipped
            - c_loss * np.log(jlim / (jlim - j_clipped - jleak))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]
        e0 = self.options["reversible_electric_potential"]
        ds = self.options["entropy_difference"]
        gas_const = self.options["gas_constant"]
        faraday_const = self.options["faraday_constant"]
        t0 = np.full(number_of_points, self.options["standard_temperature"])
        temp_op = inputs["operating_temperature"]
        a_transfer = self.options["cathode_transfer_coefficient"]
        resistance = self.options["area_specific_resistance"]
        c_loss = self.options["mass_transport_loss_constant"]
        jlim = np.full(number_of_points, self.options["max_current_density"] * 10000)
        jleak = np.full(number_of_points, self.options["leakage_current_density"])
        p_o2 = inputs["operating_pressure"]
        pvc = inputs["ambient_pressure_voltage_correction"]

        p_h2 = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ]

        j_clipped = np.clip(
            inputs["fc_current_density"], 10.0, self.options["max_current_density"] * 10000
        )

        partials_j = np.where(
            inputs["fc_current_density"] == j_clipped,
            pvc
            * (
                -gas_const * temp_op / (2 * faraday_const * a_transfer * (j_clipped + jleak))
                - c_loss / (-j_clipped + jlim - jleak)
                - resistance * np.ones(number_of_points)
            ),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials["single_layer_pemfc_voltage", "operating_temperature"] = pvc * (
            -ds / (2 * faraday_const) * np.ones(number_of_points)
            + gas_const / (2 * faraday_const) * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - gas_const / (2 * a_transfer * faraday_const) * np.log(j_clipped + jleak)
        )

        partials["single_layer_pemfc_voltage", "operating_pressure"] = pvc * (
            gas_const * temp_op / (4 * faraday_const * p_o2)
        )

        partials[
            "single_layer_pemfc_voltage",
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
        ] = pvc * (gas_const * temp_op / (2 * faraday_const * p_h2))

        partials["single_layer_pemfc_voltage", "ambient_pressure_voltage_correction"] = (
            e0
            - ds / (2 * faraday_const) * (temp_op - t0)
            + gas_const * temp_op / (2 * faraday_const) * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - gas_const * temp_op / (2 * a_transfer * faraday_const) * np.log(j_clipped + jleak)
            - resistance * j_clipped
            - c_loss * np.log(jlim / (jlim - j_clipped - jleak))
        )

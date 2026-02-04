# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import (
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    REVERSIBLE_ELECTRIC_POTENTIAL,
    MAX_CURRENT_DENSITY_EMPIRICAL,
    MAX_CURRENT_DENSITY_ANALYTICAL,
    DEFAULT_PRESSURE_ATM,
    DEFAULT_LAYER_VOLTAGE,
)


class PerformancesPEMFCStackPolarizationCurveEmpirical(om.ExplicitComponent):
    """
    Computation of the PEMFC polarization curve. This model is based on Aerostack Ultralight 200W
    PEMFC. Details can be found in :cite:`hoogendoorn:2018`.
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
            desc="open_circuit_voltage of the single-layered PEMFC [V]",
        )

        self.options.declare(
            "activation_loss_coefficient",
            default=0.014,
            desc="activation loss coefficient of the single-layered PEMFC [V/ln(A/cm**2)]",
        )

        self.options.declare(
            "coefficient_in_concentration_loss",
            default=5.63 * 10**-6,
            desc="coefficient in concentration loss of the single-layered PEMFC [V]",
        )

        self.options.declare(
            "exponential_coefficient_in_concentration_loss",
            default=11.42,
            desc="exponential coefficient in concentration loss of the single-layered PEMFC [cm**2/A]",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "fc_current_density",
            units="A/cm**2",
            val=np.full(number_of_points, np.nan),
            desc="Current density of the PEMFC stack",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":nominal_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
            desc="The nominal pressure at which the PEMFC stack operates.",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance",
            units="ohm*cm**2",
            val=0.24,
            desc="Area ohmic resistance of the single-layered PEMFC",
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

        self.declare_partials(
            of="single_layer_pemfc_voltage",
            wrt=["fc_current_density", "operating_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="single_layer_pemfc_voltage",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":nominal_pressure",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":area_ohmic_resistance",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        ocv = self.options["open_circuit_voltage"]
        active_loss_coeff = self.options["activation_loss_coefficient"]
        resistance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance"
        ]
        m_loss = self.options["coefficient_in_concentration_loss"]
        n_loss = self.options["exponential_coefficient_in_concentration_loss"]

        i_clipped = np.clip(inputs["fc_current_density"], 1e-3, MAX_CURRENT_DENSITY_EMPIRICAL)

        operating_pressure = inputs["operating_pressure"]

        nominal_pressure = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":nominal_pressure"
        ]

        pressure_ratio_log = np.log(operating_pressure / nominal_pressure)

        pressure_coeff = -0.0032 * pressure_ratio_log**2 + 0.0019 * pressure_ratio_log + 0.0542

        outputs["single_layer_pemfc_voltage"] = (
            ocv
            - active_loss_coeff * np.log(i_clipped)
            - resistance * i_clipped
            - m_loss * np.exp(n_loss * i_clipped)
            + pressure_coeff * pressure_ratio_log
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        active_loss_coeff = self.options["activation_loss_coefficient"]
        resistance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance"
        ]
        m_loss = self.options["coefficient_in_concentration_loss"]
        n_loss = self.options["exponential_coefficient_in_concentration_loss"]

        operating_pressure = inputs["operating_pressure"]

        nominal_pressure = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":nominal_pressure"
        ]

        pressure_ratio_log = np.log(operating_pressure / nominal_pressure)

        i_clipped = np.clip(inputs["fc_current_density"], 1e-3, MAX_CURRENT_DENSITY_EMPIRICAL)

        partials_j = np.where(
            inputs["fc_current_density"] == i_clipped,
            -active_loss_coeff / i_clipped
            - resistance
            - m_loss * n_loss * np.exp(n_loss * i_clipped),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials[
            "single_layer_pemfc_voltage",
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance",
        ] = -i_clipped

        partials["single_layer_pemfc_voltage", "operating_pressure"] = -(
            48 * pressure_ratio_log**2 - 19 * pressure_ratio_log - 271
        ) / (5000 * operating_pressure)

        partials[
            "single_layer_pemfc_voltage",
            "data:propulsion:he_power_train:PEMFC_stack:" + pemfc_stack_id + ":nominal_pressure",
        ] = (48 * pressure_ratio_log**2 - 19 * pressure_ratio_log - 271) / (5000 * nominal_pressure)


class PerformancesPEMFCStackPolarizationCurveAnalytical(om.ExplicitComponent):
    """
    Computation of the single layer voltage of the PEMFC.This model is based on analytical i-v
    curve equation. Details can be found in :cite:`Juschus:2021`. The hydrogen oxidation reaction
    is assumed to produce liquid water following :cite:`baschuk:2005`.
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
            "entropy_difference",
            default=163.23,
            desc="entropy difference from reaction producing liquid H2O in [J/(mol*K)]",
        )
        self.options.declare(
            "standard_temperature",
            default=298.15,
            desc="standard temperature for the hydrogen oxidation reaction in [K]",
        )
        self.options.declare(
            "operating_temperature",
            default=350.0,
            desc="standard operating temperature for the PEMFC in [K]",
        )
        self.options.declare(
            "cathode_transfer_coefficient",
            default=0.3,
            desc="transfer coefficient at the cathode side of fuel cell",
        )
        self.options.declare(
            "mass_transport_loss_constant",
            default=0.1,
            desc="the constant result from mass transport in the PEMFC [V]",
        )
        self.options.declare(
            "leakage_current_density",
            default=100.0,
            desc="leak loss of current density from the PEMFC [A/m**2]",
        )

    def setup(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "fc_current_density",
            units="A/m**2",
            val=np.full(number_of_points, np.nan),
            desc="Current density of the PEMFC stack",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
            units="atm",
            val=DEFAULT_PRESSURE_ATM,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance",
            units="ohm*m**2",
            val=1e-6,
            desc="Area ohmic resistance of the single-layered PEMFC",
        )
        self.add_input(
            "operating_pressure",
            units="atm",
            val=np.full(number_of_points, DEFAULT_PRESSURE_ATM),
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

    def setup_partials(self):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=[
                "fc_current_density",
                "operating_pressure",
                "ambient_pressure_voltage_correction",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":hydrogen_reactant_pressure",
                "data:propulsion:he_power_train:PEMFC_stack:"
                + pemfc_stack_id
                + ":area_ohmic_resistance",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        e0 = REVERSIBLE_ELECTRIC_POTENTIAL
        ds = self.options["entropy_difference"]
        t_0 = np.full(number_of_points, self.options["standard_temperature"])
        a_transfer = self.options["cathode_transfer_coefficient"]
        resistance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance"
        ]
        c_loss = self.options["mass_transport_loss_constant"]
        j_lim = np.full(number_of_points, MAX_CURRENT_DENSITY_ANALYTICAL * 10000.0)
        j_leak = np.full(number_of_points, self.options["leakage_current_density"])
        pvc = inputs["ambient_pressure_voltage_correction"]
        j_clipped = np.clip(
            inputs["fc_current_density"],
            10.0,
            0.99
            * (MAX_CURRENT_DENSITY_ANALYTICAL * 10000.0 - self.options["leakage_current_density"]),
        )
        p_o2 = inputs["operating_pressure"]
        p_h2 = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ]
        t_operating = self.options["operating_temperature"]

        outputs["single_layer_pemfc_voltage"] = pvc * (
            e0
            - ds / (2.0 * FARADAY_CONSTANT) * (t_operating - t_0)
            + GAS_CONSTANT
            * t_operating
            / (2.0 * FARADAY_CONSTANT)
            * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - GAS_CONSTANT
            * t_operating
            / (2.0 * a_transfer * FARADAY_CONSTANT)
            * np.log(j_clipped + j_leak)
            - resistance * j_clipped
            - c_loss * np.log(j_lim / (j_lim - j_clipped - j_leak))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        e0 = REVERSIBLE_ELECTRIC_POTENTIAL
        ds = self.options["entropy_difference"]
        t_0 = np.full(number_of_points, self.options["standard_temperature"])
        t_operating = self.options["operating_temperature"]
        a_transfer = self.options["cathode_transfer_coefficient"]
        resistance = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance"
        ]
        c_loss = self.options["mass_transport_loss_constant"]
        j_lim = np.full(number_of_points, MAX_CURRENT_DENSITY_ANALYTICAL * 10000.0)
        j_leak = np.full(number_of_points, self.options["leakage_current_density"])
        p_o2 = inputs["operating_pressure"]
        pvc = inputs["ambient_pressure_voltage_correction"]
        p_h2 = inputs[
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure"
        ]
        j_clipped = np.clip(
            inputs["fc_current_density"],
            10.0,
            0.99
            * (MAX_CURRENT_DENSITY_ANALYTICAL * 10000.0 - self.options["leakage_current_density"]),
        )

        partials_j = np.where(
            inputs["fc_current_density"] == j_clipped,
            pvc
            * (
                -GAS_CONSTANT
                * t_operating
                / (2.0 * FARADAY_CONSTANT * a_transfer * (j_clipped + j_leak))
                - c_loss / (-j_clipped + j_lim - j_leak)
                - np.full(number_of_points, resistance)
            ),
            1e-6,
        )

        partials["single_layer_pemfc_voltage", "fc_current_density"] = partials_j

        partials[
            "single_layer_pemfc_voltage",
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":area_ohmic_resistance",
        ] = -j_clipped

        partials["single_layer_pemfc_voltage", "operating_pressure"] = pvc * (
            GAS_CONSTANT * t_operating / (4.0 * FARADAY_CONSTANT * p_o2)
        )

        partials[
            "single_layer_pemfc_voltage",
            "data:propulsion:he_power_train:PEMFC_stack:"
            + pemfc_stack_id
            + ":hydrogen_reactant_pressure",
        ] = pvc * (GAS_CONSTANT * t_operating / (2.0 * FARADAY_CONSTANT * p_h2))

        partials["single_layer_pemfc_voltage", "ambient_pressure_voltage_correction"] = (
            e0
            - ds / (2.0 * FARADAY_CONSTANT) * (t_operating - t_0)
            + GAS_CONSTANT
            * t_operating
            / (2.0 * FARADAY_CONSTANT)
            * np.log(p_h2 * np.sqrt(p_o2 * 0.21))
            - GAS_CONSTANT
            * t_operating
            / (2.0 * a_transfer * FARADAY_CONSTANT)
            * np.log(j_clipped + j_leak)
            - resistance * j_clipped
            - c_loss * np.log(j_lim / (j_lim - j_clipped - j_leak))
        )

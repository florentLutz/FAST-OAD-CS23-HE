# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingDCSSPCEfficiency(om.ExplicitComponent):
    """
    It was observed that having a varying efficiency for the SSPC, even for very small variation
    can cause the process not to converge. It was thus decided to have a fixed efficiency during
    the whole mission. That being said, this efficiency depends on the power caliber see
    :cite:`valente:2021` and we know that this efficiency mainly depends on the losses in the
    current conducting part of the SSPC which is approximated as 2 IGBT modules and diode see
    :cite:`liu:2012`. Default value for gate voltage are taken from the reference IGBT, the
    SEMiX453GB12M7p.
    """

    def initialize(self):
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of the SSPC",
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
            units="V",
            val=np.nan,
            desc="Voltage caliber of the SSPC",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:gate_voltage",
            units="V",
            val=0.87,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:gate_voltage",
            units="V",
            val=1.3,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance",
            val=np.nan,
            units="ohm",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            val=0.99,
            desc="Value of the SSPC efficiency, assumed constant during operations (eases convergence)",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"
        ]

        gate_voltage = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:gate_voltage"]
            + inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:gate_voltage"]
        )
        module_level_resistance = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance"]
            + inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance"]
        )

        losses = gate_voltage * current_caliber + module_level_resistance * current_caliber**2.0
        power_caliber = voltage_caliber * current_caliber

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency"] = (
            power_caliber - losses
        ) / power_caliber

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_sspc_id = self.options["dc_sspc_id"]

        current_caliber = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber"
        ]
        voltage_caliber = inputs[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber"
        ]

        gate_voltage = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:gate_voltage"]
            + inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:gate_voltage"]
        )
        module_level_resistance = (
            inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance"]
            + inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance"]
        )

        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_caliber",
        ] = -module_level_resistance / voltage_caliber
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_caliber",
        ] = (gate_voltage + module_level_resistance * current_caliber) / voltage_caliber**2.0
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:gate_voltage",
        ] = -1.0 / voltage_caliber
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:gate_voltage",
        ] = -1.0 / voltage_caliber
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance",
        ] = -current_caliber / voltage_caliber
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance",
        ] = -current_caliber / voltage_caliber

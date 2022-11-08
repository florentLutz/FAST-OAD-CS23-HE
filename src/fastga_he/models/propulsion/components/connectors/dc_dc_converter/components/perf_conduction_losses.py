# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesConductionLosses(om.ExplicitComponent):
    """Computation of Conduction losses for the IGBT and the diode."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("duty_cycle", val=np.full(number_of_points, np.nan))
        self.add_input("current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance",
            units="ohm",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:resistance",
            units="ohm",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:gate_voltage",
            units="V",
            val=np.nan,
        )

        self.add_output(
            "conduction_losses_diode",
            units="W",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
        )
        self.add_output(
            "conduction_losses_IGBT",
            units="W",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
        )

        self.declare_partials(
            of="conduction_losses_diode",
            wrt=[
                "duty_cycle",
                "current_out",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:resistance",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":diode:gate_voltage",
            ],
        )
        self.declare_partials(
            of="conduction_losses_IGBT",
            wrt=[
                "duty_cycle",
                "current_out",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:resistance",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        v_d0 = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:gate_voltage"
        ]

        r_igbt = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance"
        ]
        r_d = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:resistance"
        ]

        duty_cycle = inputs["duty_cycle"]
        current_out = inputs["current_out"]

        conduction_loss_igbt = duty_cycle / (1.0 - duty_cycle) ** 2.0 * r_igbt * current_out ** 2.0
        conduction_loss_diode = (
            current_out * v_d0 + 1.0 / (1.0 - duty_cycle) * r_d * current_out ** 2.0
        )

        outputs["conduction_losses_diode"] = conduction_loss_diode
        outputs["conduction_losses_IGBT"] = conduction_loss_igbt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        v_d0 = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:gate_voltage"
        ]

        r_igbt = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance"
        ]
        r_d = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:resistance"
        ]

        duty_cycle = inputs["duty_cycle"]
        current_out = inputs["current_out"]

        partials[
            "conduction_losses_diode",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:gate_voltage",
        ] = current_out
        partials[
            "conduction_losses_diode",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:resistance",
        ] = (
            1.0 / (1.0 - duty_cycle) * current_out ** 2.0
        )
        partials[
            "conduction_losses_diode",
            "duty_cycle",
        ] = np.diag(1.0 / (1.0 - duty_cycle) ** 2.0 * r_d * current_out ** 2.0)
        partials["conduction_losses_diode", "current_out"] = np.diag(
            v_d0 + 2.0 / (1.0 - duty_cycle) * r_d * current_out
        )

        partials[
            "conduction_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance",
        ] = (
            duty_cycle / (1.0 - duty_cycle) ** 2.0 * current_out ** 2.0
        )
        partials["conduction_losses_IGBT", "duty_cycle"] = np.diag(
            (1.0 - duty_cycle ** 2.0) / (1.0 - duty_cycle) ** 4.0 * r_igbt * current_out ** 2.0
        )
        partials["conduction_losses_IGBT", "current_out"] = np.diag(
            2.0 * duty_cycle / (1.0 - duty_cycle) ** 2.0 * r_igbt * current_out
        )

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
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):

        inverted_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("modulation_index", val=np.full(number_of_points, np.nan))
        self.add_input("current", units="A", val=np.full(number_of_points, np.nan))

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:resistance",
            units="ohm",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:gate_voltage",
            units="V",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:resistance",
            units="ohm",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:gate_voltage",
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
                "modulation_index",
                "current",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:resistance",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:gate_voltage",
            ],
        )
        self.declare_partials(
            of="conduction_losses_IGBT",
            wrt=[
                "modulation_index",
                "current",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:resistance",
                "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:gate_voltage",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverted_id = self.options["inverter_id"]
        cos_phi = inputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor"]
        v_ce0 = inputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:gate_voltage"
        ]
        v_d0 = inputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:gate_voltage"
        ]
        r_igbt = inputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:resistance"
        ]
        r_d = inputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:resistance"]

        beta = inputs["modulation_index"]
        current = inputs["current"]

        conduction_loss_diode = v_d0 * current / (2.0 * np.pi) * (
            1.0 - np.pi / 4.0 * beta * cos_phi
        ) + r_d * current ** 2.0 / 8.0 * (1.0 - 8.0 / (3.0 * np.pi) * beta * cos_phi)
        conduction_loss_igbt = v_ce0 * current / (2.0 * np.pi) * (
            1.0 + np.pi / 4.0 * beta * cos_phi
        ) + r_igbt * current ** 2.0 / 8.0 * (1.0 + 8.0 / (3.0 * np.pi) * beta * cos_phi)

        outputs["conduction_losses_diode"] = conduction_loss_diode
        outputs["conduction_losses_IGBT"] = conduction_loss_igbt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverted_id = self.options["inverter_id"]

        cos_phi = inputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor"]
        v_ce0 = inputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:gate_voltage"
        ]
        v_d0 = inputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:gate_voltage"
        ]
        r_igbt = inputs[
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:resistance"
        ]
        r_d = inputs["data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:resistance"]

        beta = inputs["modulation_index"]
        current = inputs["current"]

        partials[
            "conduction_losses_diode",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor",
        ] = -beta * (v_d0 * current / 8.0 + r_d * current ** 2.0 / (3.0 * np.pi))
        partials[
            "conduction_losses_diode",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:gate_voltage",
        ] = (
            current / (2.0 * np.pi) * (1.0 - np.pi / 4.0 * beta * cos_phi)
        )
        partials[
            "conduction_losses_diode",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":diode:resistance",
        ] = (
            current ** 2.0 / 8.0 * (1.0 - 8.0 / (3.0 * np.pi) * beta * cos_phi)
        )
        partials[
            "conduction_losses_diode",
            "modulation_index",
        ] = -np.diag(cos_phi * (v_d0 * current / 8.0 + r_d * current ** 2.0 / (3.0 * np.pi)))
        partials["conduction_losses_diode", "current"] = np.diag(
            v_d0 / (2.0 * np.pi) * (1.0 - np.pi / 4.0 * beta * cos_phi)
            + r_d * current / 4.0 * (1.0 - 8.0 / (3.0 * np.pi) * beta * cos_phi)
        )

        partials[
            "conduction_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":power_factor",
        ] = (
            v_ce0 * current / 8.0 * beta + r_igbt * current ** 2.0 / (3.0 * np.pi) * beta
        )
        partials[
            "conduction_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:gate_voltage",
        ] = (
            current / (2.0 * np.pi) * (1.0 + np.pi / 4.0 * beta * cos_phi)
        )
        partials[
            "conduction_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverted_id + ":igbt:resistance",
        ] = (
            current ** 2.0 / 8.0 * (1.0 + 8.0 / (3.0 * np.pi) * beta * cos_phi)
        )
        partials["conduction_losses_IGBT", "modulation_index"] = np.diag(
            v_ce0 * current / 8.0 * cos_phi + r_igbt * current ** 2.0 / (3.0 * np.pi) * cos_phi
        )
        partials["conduction_losses_IGBT", "current"] = np.diag(
            v_ce0 / (2.0 * np.pi) * (1.0 + np.pi / 4.0 * beta * cos_phi)
            + r_igbt * current / 4.0 * (1.0 + 8.0 / (3.0 * np.pi) * beta * cos_phi)
        )
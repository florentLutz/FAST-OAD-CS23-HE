# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesSwitchingLosses(om.ExplicitComponent):
    """Computation of switching losses for the IGBT and the diode."""

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

        inverter_id = self.options["inverter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("switching_frequency", units="Hz", val=np.full(number_of_points, np.nan))
        self.add_input("current", units="A", val=np.full(number_of_points, np.nan))

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c",
            val=np.nan,
        )

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c",
            val=np.nan,
        )

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c",
            val=np.nan,
        )

        self.add_output(
            "switching_losses_diode",
            units="W",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
        )
        self.add_output(
            "switching_losses_IGBT",
            units="W",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
        )

        self.declare_partials(
            of="switching_losses_diode",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c",
                "switching_frequency",
                "current",
            ],
        )

        self.declare_partials(
            of="switching_losses_IGBT",
            wrt=[
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b",
                "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c",
                "switching_frequency",
                "current",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        inverter_id = self.options["inverter_id"]

        a_on = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a"]
        b_on = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b"]
        c_on = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c"]

        a_off = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a"]
        b_off = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b"]
        c_off = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c"]

        a_rr = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a"]
        b_rr = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b"]
        c_rr = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c"]

        f_sw = inputs["switching_frequency"]
        current = inputs["current"]

        loss_diode = f_sw * (a_rr / 2.0 + b_rr * current / np.pi + c_rr * current ** 2.0 / 4)
        loss_igbt = f_sw * (
            (a_on + a_off) / 2.0
            + (b_on + b_off) * current / np.pi
            + (c_on + c_off) * current ** 2.0 / 4
        )
        # a, b and c coefficient on reference were interpolated to give the results in J

        outputs["switching_losses_diode"] = loss_diode
        outputs["switching_losses_IGBT"] = loss_igbt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        a_on = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a"]
        b_on = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b"]
        c_on = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c"]

        a_off = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a"]
        b_off = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b"]
        c_off = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c"]

        a_rr = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a"]
        b_rr = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b"]
        c_rr = inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c"]

        f_sw = inputs["switching_frequency"]
        current = inputs["current"]

        partials[
            "switching_losses_diode",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:a",
        ] = (
            f_sw / 2.0 * 1e-3
        )
        partials[
            "switching_losses_diode",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:b",
        ] = (
            f_sw * current / np.pi * 1e-3
        )
        partials[
            "switching_losses_diode",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_rr:c",
        ] = (
            f_sw * current ** 2.0 / 4 * 1e-3
        )
        partials["switching_losses_diode", "switching_frequency"] = (
            np.diag(a_rr / 2.0 + b_rr * current / np.pi + c_rr * current ** 2.0 / 4) * 1e-3
        )
        partials["switching_losses_diode", "current"] = np.diag(
            f_sw * (b_rr / np.pi + c_rr * current / 2.0) * 1e-3
        )

        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:a",
        ] = (
            f_sw / 2.0 * 1e-3
        )
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:a",
        ] = (
            f_sw / 2.0 * 1e-3
        )
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:b",
        ] = (
            f_sw * current / np.pi * 1e-3
        )
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:b",
        ] = (
            f_sw * current / np.pi * 1e-3
        )
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_on:c",
        ] = (
            f_sw * current ** 2.0 / 4 * 1e-3
        )
        partials[
            "switching_losses_IGBT",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":energy_off:c",
        ] = (
            f_sw * current ** 2.0 / 4 * 1e-3
        )
        partials["switching_losses_IGBT", "switching_frequency"] = (
            np.diag(
                (a_on + a_off) / 2.0
                + (b_on + b_off) * current / np.pi
                + (c_on + c_off) * current ** 2.0 / 4
            )
            * 1e-3
        )
        partials["switching_losses_IGBT", "current"] = (
            np.diag(f_sw * ((b_on + b_off) / np.pi + (c_on + c_off) * current / 2.0)) * 1e-3
        )

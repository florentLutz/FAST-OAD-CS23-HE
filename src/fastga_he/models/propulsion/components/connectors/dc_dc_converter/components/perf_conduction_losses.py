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

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_input(
            "current_IGBT",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current going through the switch",
        )
        self.add_input(
            "current_capacitor",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current going through the filter capacitor",
        )
        self.add_input(
            "current_diode",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current going through the diode",
        )
        self.add_input(
            "current_inductor",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current going through the inductor",
        )

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
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
            units="ohm",
            val=1.4e-3,
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:resistance",
            units="ohm",
            val=1.4e-3,
        )

        self.add_output(
            "conduction_losses_diode",
            units="W",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=np.full(number_of_points, 0.0),
        )
        self.add_output(
            "conduction_losses_IGBT",
            units="W",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=np.full(number_of_points, 0.0),
        )
        self.add_output(
            "conduction_losses_inductor",
            units="W",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=np.full(number_of_points, 0.0),
        )
        self.add_output(
            "conduction_losses_capacitor",
            units="W",
            val=np.full(number_of_points, 1.0),
            shape=number_of_points,
            lower=np.full(number_of_points, 0.0),
        )

        self.declare_partials(
            of="conduction_losses_diode",
            wrt=[
                "dc_current_out",
                "current_diode",
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
                "current_IGBT",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":igbt:resistance",
            ],
        )
        self.declare_partials(
            of="conduction_losses_inductor",
            wrt=[
                "current_inductor",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":inductor:resistance",
            ],
        )
        self.declare_partials(
            of="conduction_losses_capacitor",
            wrt=[
                "current_capacitor",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":capacitor:resistance",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        r_igbt = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance"
        ]
        i_igbt = inputs["current_IGBT"]

        r_d = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:resistance"
        ]
        v_d0 = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:gate_voltage"
        ]
        i_d = inputs["current_diode"]
        current_out = inputs["dc_current_out"]

        r_inductor = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance"
        ]
        i_inductor = inputs["current_inductor"]

        r_capacitor = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:resistance"
        ]
        i_capacitor = inputs["current_capacitor"]

        conduction_loss_igbt = r_igbt * i_igbt ** 2.0
        conduction_loss_diode = current_out * v_d0 + r_d * i_d ** 2.0
        conduction_losses_inductor = i_inductor ** 2.0 * r_inductor
        conduction_losses_capacitor = i_capacitor ** 2.0 * r_capacitor

        outputs["conduction_losses_diode"] = conduction_loss_diode
        outputs["conduction_losses_IGBT"] = conduction_loss_igbt
        outputs["conduction_losses_inductor"] = conduction_losses_inductor
        outputs["conduction_losses_capacitor"] = conduction_losses_capacitor

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        r_igbt = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance"
        ]
        i_igbt = inputs["current_IGBT"]

        r_d = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:resistance"
        ]
        v_d0 = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:gate_voltage"
        ]
        i_d = inputs["current_diode"]
        current_out = inputs["dc_current_out"]

        r_inductor = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance"
        ]
        i_inductor = inputs["current_inductor"]

        r_capacitor = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:resistance"
        ]
        i_capacitor = inputs["current_capacitor"]

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
            i_d ** 2.0
        )
        partials["conduction_losses_diode", "current_diode"] = np.diag(2.0 * r_d * i_d)
        partials["conduction_losses_diode", "dc_current_out"] = np.eye(number_of_points) * v_d0

        partials[
            "conduction_losses_IGBT",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:resistance",
        ] = (
            i_igbt ** 2.0
        )
        partials["conduction_losses_IGBT", "current_IGBT"] = np.diag(2.0 * r_igbt * i_igbt)

        partials[
            "conduction_losses_inductor",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:resistance",
        ] = (
            i_inductor ** 2.0
        )
        partials["conduction_losses_inductor", "current_inductor"] = np.diag(
            2.0 * r_inductor * i_inductor
        )

        partials[
            "conduction_losses_capacitor",
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:resistance",
        ] = (
            i_capacitor ** 2.0
        )
        partials["conduction_losses_capacitor", "current_capacitor"] = np.diag(
            2.0 * r_capacitor * i_capacitor
        )

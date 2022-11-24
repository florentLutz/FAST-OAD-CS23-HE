# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum current going into each component and the maximum voltage at
    the input and output of the converter.
    """

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

        self.add_input(
            "dc_current_in",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the input side of the converter",
        )
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
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the input side of the converter",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage to output side",
        )
        self.add_input(
            "losses_converter",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max",
            units="A",
            val=500.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max",
            wrt="dc_current_in",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:current_max",
            units="A",
            val=500.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:current_max",
            wrt="current_IGBT",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:current_max",
            units="A",
            val=500.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:current_max",
            wrt="current_diode",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max",
            units="A",
            val=500.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max",
            wrt="current_inductor",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max",
            units="A",
            val=500.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max",
            wrt="current_capacitor",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max",
            units="V",
            val=500,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max",
            wrt="dc_voltage_in",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_max",
            units="V",
            val=500,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_max",
            wrt="dc_voltage_out",
            method="exact",
        )

        self.add_output(
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":losses_max",
            units="W",
            val=500,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":losses_max",
            wrt="losses_converter",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max"
        ] = np.max(inputs["dc_current_in"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:current_max"
        ] = np.max(inputs["current_IGBT"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:current_max"
        ] = np.max(inputs["current_diode"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max"
        ] = np.max(inputs["current_inductor"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max"
        ] = np.max(inputs["current_capacitor"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max"
        ] = np.max(inputs["dc_voltage_in"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_max"
        ] = np.max(inputs["dc_voltage_out"])

        outputs[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":losses_max"
        ] = np.max(inputs["losses_converter"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_dc_converter_id = self.options["dc_dc_converter_id"]

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":current_in_max",
            "dc_current_in",
        ] = np.where(inputs["dc_current_in"] == np.max(inputs["dc_current_in"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":igbt:current_max",
            "current_IGBT",
        ] = np.where(inputs["current_IGBT"] == np.max(inputs["current_IGBT"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":diode:current_max",
            "current_diode",
        ] = np.where(inputs["current_diode"] == np.max(inputs["current_diode"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":inductor:current_max",
            "current_inductor",
        ] = np.where(inputs["current_inductor"] == np.max(inputs["current_inductor"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":capacitor:current_max",
            "current_capacitor",
        ] = np.where(inputs["current_capacitor"] == np.max(inputs["current_capacitor"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_in_max",
            "dc_voltage_in",
        ] = np.where(inputs["dc_voltage_in"] == np.max(inputs["dc_voltage_in"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":voltage_out_max",
            "dc_voltage_out",
        ] = np.where(inputs["dc_voltage_out"] == np.max(inputs["dc_voltage_out"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:DC_DC_converter:" + dc_dc_converter_id + ":losses_max",
            "losses_converter",
        ] = np.where(inputs["losses_converter"] == np.max(inputs["losses_converter"]), 1.0, 0.0)

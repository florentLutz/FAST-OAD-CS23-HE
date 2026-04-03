# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

DEFAULT_STACK_VOLTAGE = 325.0  # [V]


class PerformancesPEMFCStackBOPVoltage(om.ExplicitComponent):
    """
    Output voltage computation of the PEMFC stack, assumes for now that it is equal to
    the voltage output of the modules but may change in the future. This is why it is in a separate
    module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # By default this is the name of the output of this component, however, depending on the
        # mode, we might want to change it
        self.output_name = "voltage_out"

    def initialize(self):
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the PEMFC stack is directly connected to a bus, a special mode is required to "
            "interface the two",
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]

        if self.options["direct_bus_connection"]:
            self.output_name = "pemfc_voltage"

        self.add_input(
            "single_layer_pemfc_voltage", units="V", val=np.full(number_of_points, np.nan)
        )

        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
            val=np.nan,
            desc="Total number of layers in the PEMFC stack",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
            units="W",
            val=0.0,
            shape=number_of_points,
        )
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_output(
            self.output_name, units="V", val=np.full(number_of_points, DEFAULT_STACK_VOLTAGE)
        )
        self.add_output(
            "fuel_cell_voltage", units="V", val=np.full(number_of_points, DEFAULT_STACK_VOLTAGE)
        )

        self.declare_partials(
            of=self.output_name,
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="fuel_cell_voltage",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of=self.output_name,
            wrt=[
                "single_layer_pemfc_voltage",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":bop_power_required",
                "dc_current_out",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="fuel_cell_voltage",
            wrt=[
                "single_layer_pemfc_voltage",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":bop_power_required",
                "dc_current_out",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        single_layer_voltage = inputs["single_layer_pemfc_voltage"]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers"
        ]
        bop_power_required = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required"
        ]
        dc_current_out = inputs["dc_current_out"]

        unclipped_voltage = (
            single_layer_voltage * number_of_layers - bop_power_required / dc_current_out
        )
        unclipped_fuel_cell_voltage = single_layer_voltage * number_of_layers

        outputs[self.output_name] = np.clip(unclipped_voltage, 5.0, np.inf)
        outputs["fuel_cell_voltage"] = np.clip(unclipped_fuel_cell_voltage, 5.0, np.inf)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]

        single_layer_voltage = inputs["single_layer_pemfc_voltage"]
        number_of_layers = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers"
        ]
        bop_power_required = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required"
        ]
        dc_current_out = inputs["dc_current_out"]

        unclipped_voltage = (
            single_layer_voltage * number_of_layers - bop_power_required / dc_current_out
        )
        unclipped_fuel_cell_voltage = single_layer_voltage * number_of_layers

        clipped_voltage = np.clip(unclipped_voltage, 5.0, np.inf)
        clipped_fuel_cell_voltage = np.clip(unclipped_fuel_cell_voltage, 5.0, np.inf)

        partials[self.output_name, "single_layer_pemfc_voltage"] = np.where(
            clipped_voltage == unclipped_voltage, np.full(number_of_points, number_of_layers), 1e-6
        )

        partials[
            self.output_name,
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
        ] = np.where(clipped_voltage == unclipped_voltage, single_layer_voltage, 1e-6)

        partials[
            self.output_name,
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":bop_power_required",
        ] = np.where(clipped_voltage == unclipped_voltage, -1.0 / dc_current_out, 1e-6)

        partials[self.output_name, "dc_current_out"] = np.where(
            clipped_voltage == unclipped_voltage, bop_power_required / dc_current_out**2.0, 1e-6
        )

        partials["fuel_cell_voltage", "single_layer_pemfc_voltage"] = np.where(
            clipped_fuel_cell_voltage == unclipped_fuel_cell_voltage,
            np.full(number_of_points, number_of_layers),
            1e-6,
        )

        partials[
            "fuel_cell_voltage",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":number_of_layers",
        ] = np.where(
            clipped_fuel_cell_voltage == unclipped_fuel_cell_voltage, single_layer_voltage, 1e-6
        )

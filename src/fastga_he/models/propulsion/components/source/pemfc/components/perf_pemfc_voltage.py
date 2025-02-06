# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

DEFAULT_STACK_VOLTAGE = 325.0


class PerformancesPEMFCVoltage(om.ExplicitComponent):
    """
    Computation of the voltage at the output of the battery, assumes for now that it is equal to
    the voltage output of the modules. May change in the future hence why it is in a separate
    module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # By default this is the name of the output of this component, however, depending on the
        # mode, we might want to change it
        self.output_name = "voltage_out"

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="direct_bus_connection",
            default=False,
            types=bool,
            desc="If the battery is directly connected to a bus, a special mode is required to "
            "interface the two",
        )
        self.options.declare(
            name="pemfc_stack_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        pemfc_stack_id = self.options["pemfc_stack_id"]

        if self.options["direct_bus_connection"]:
            self.output_name = "pemfc_voltage"

        self.add_input(
            "single_layer_pemfc_voltage", units="V", val=np.full(number_of_points, np.nan)
        )

        self.add_input(
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
            val=np.nan,
            desc="Total number of layers in the pemfc stacks",
        )

        self.add_output(
            self.output_name, units="V", val=np.full(number_of_points, DEFAULT_STACK_VOLTAGE)
        )

        self.declare_partials(
            of=self.output_name,
            wrt="data:propulsion:he_power_train:pemfc_stack:"
            + pemfc_stack_id
            + ":number_of_layers",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of=self.output_name,
            wrt="single_layer_pemfc_voltage",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]

        outputs[self.output_name] = (
            inputs["single_layer_pemfc_voltage"]
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_id = self.options["pemfc_stack_id"]
        number_of_points = self.options["number_of_points"]

        partials[self.output_name, "single_layer_pemfc_voltage"] = (
            np.ones(number_of_points)
            * inputs[
                "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers"
            ]
        )

        partials[
            self.output_name,
            "data:propulsion:he_power_train:pemfc_stack:" + pemfc_stack_id + ":number_of_layers",
        ] = inputs["single_layer_pemfc_voltage"]

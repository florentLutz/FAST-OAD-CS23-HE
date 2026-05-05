# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleDrag(om.ExplicitComponent):
    """
    Computation of the nozzle drag.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="pemfc_stack_bop_id",
            default=None,
            desc="Identifier of the PEMFC stack",
            allow_none=False,
        )
        self.options.declare(
            name="nozzle_id",
            default=None,
            desc="Identifier of the diffuser",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.add_input(
            "exit_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "ambient_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )
        self.add_input(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
            val=np.nan,
            units="m**2",
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            val=0.0,
            units="N",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        self.declare_partials(
            of="*",
            wrt=["exit_pressure", "ambient_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        exit_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area"
        ]
        pressure_difference = inputs["ambient_pressure"] - inputs["exit_pressure"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag"
        ] = np.clip(exit_area * pressure_difference, 0.0, np.inf)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        exit_area = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area"
        ]
        exit_pressure = inputs["exit_pressure"]
        ambient_pressure = inputs["ambient_pressure"]
        drag = exit_area * (ambient_pressure - exit_pressure)
        clipped_drag = np.clip(drag, 0.0, np.inf)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            "exit_pressure",
        ] = np.where(
            drag == clipped_drag,
            np.full(
                number_of_points,
                exit_area,
            ),
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            "exit_pressure",
        ] = np.where(drag == clipped_drag, np.full(number_of_points, -exit_area), 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":exit_area",
        ] = np.where(drag == clipped_drag, ambient_pressure - exit_pressure, 1e-6)

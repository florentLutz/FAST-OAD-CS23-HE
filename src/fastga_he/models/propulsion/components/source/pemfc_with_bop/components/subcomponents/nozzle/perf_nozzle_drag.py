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
            "nozzle_exit_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "true_airspeed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            name="air_mass_flow_rate",
            val=np.nan,
            units="kg/s",
            shape=number_of_points,
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

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        unclipped_drag = inputs["air_mass_flow_rate"] * (
            inputs["true_airspeed"] - inputs["nozzle_exit_air_speed"]
        )

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag"
        ] = np.clip(unclipped_drag, -np.inf, 0.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]

        unclipped_drag = inputs["air_mass_flow_rate"] * (
            inputs["true_airspeed"] - inputs["nozzle_exit_air_speed"]
        )

        clipped_drag = np.clip(unclipped_drag, -np.inf, 0.0)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            "air_mass_flow_rate",
        ] = np.where(
            unclipped_drag == clipped_drag,
            inputs["true_airspeed"] - inputs["nozzle_exit_air_speed"],
            1e-6,
        )

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            "nozzle_exit_air_speed",
        ] = np.where(unclipped_drag == clipped_drag, -inputs["air_mass_flow_rate"], 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":drag",
            "true_airspeed",
        ] = np.where(unclipped_drag == clipped_drag, inputs["air_mass_flow_rate"], 1e-6)

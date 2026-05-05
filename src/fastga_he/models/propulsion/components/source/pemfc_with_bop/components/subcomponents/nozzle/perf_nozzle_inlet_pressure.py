# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleInletPressure(om.ExplicitComponent):
    """
    Computation of the exit air pressure from the heat exchanger.
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
            name="connected_heat_exchanger_id",
            default=None,
            desc="Identifier of the connected heat exchanger",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            units="Pa",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            name="diffuser_exit_pressure",
            units="Pa",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "ambient_pressure",
            units="Pa",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output("nozzle_inlet_pressure", val=1e5, units="Pa", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]

        self.declare_partials(
            of="*",
            wrt=["diffuser_exit_pressure", "ambient_pressure"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=0.5,
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=-0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        outputs["nozzle_inlet_pressure"] = 0.5 * (
            inputs["diffuser_exit_pressure"]
            - inputs[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":air_pressure_drop"
            ]
            + inputs["ambient_pressure"]
        )

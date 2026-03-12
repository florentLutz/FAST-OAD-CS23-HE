# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesInletPressureDrop(om.ExplicitComponent):
    """
    Computation of the total throat pressure.
    """

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
            name="air_inlet_id",
            default=None,
            desc="Identifier of the air inlet",
            allow_none=False,
        )

    def setup(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]
        air_inlet_id = self.options["air_inlet_id"]

        self.add_input("dynamic_pressure", units="Pa", val=np.full(number_of_points, np.nan))
        self.add_input(
            "throat_total_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )

        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":air_pressure_drop",
            val=1e6,
            units="Pa",
        )
        self.add_output(
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_ambient_dynamic_pressure",
            val=7200.0,
            units="Pa",
        )

    def setup_partials(self):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        number_of_points = self.options["number_of_points"]
        air_inlet_id = self.options["air_inlet_id"]

        self.declare_partials(
            of="*",
            wrt="dynamic_pressure",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":air_pressure_drop",
            wrt="*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        dynamic_pressure = inputs["dynamic_pressure"]
        throat_total_pressure = inputs["throat_total_pressure"]

        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":air_pressure_drop"
        ] = np.max(throat_total_pressure - dynamic_pressure)
        outputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_ambient_dynamic_pressure"
        ] = np.max(dynamic_pressure)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        air_inlet_id = self.options["air_inlet_id"]

        dynamic_pressure = inputs["dynamic_pressure"]
        throat_total_pressure = inputs["throat_total_pressure"]

        max_pressure_drop = np.max(throat_total_pressure - dynamic_pressure)
        max_dynamic_pressure = np.max(dynamic_pressure)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":air_pressure_drop",
            "dynamic_pressure",
        ] = np.where(throat_total_pressure - dynamic_pressure == max_pressure_drop, -1.0, 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":air_pressure_drop",
            "throat_total_pressure",
        ] = np.where(throat_total_pressure - dynamic_pressure == max_pressure_drop, 1.0, 1e-6)

        partials[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + air_inlet_id
            + ":max_ambient_dynamic_pressure",
            "dynamic_pressure",
        ] = np.where(dynamic_pressure == max_dynamic_pressure, 1.0, 1e-6)

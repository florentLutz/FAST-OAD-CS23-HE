# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesNozzleExitPressure(om.ExplicitComponent):
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
            name="nozzle_id",
            default=None,
            desc="Identifier of the diffuser",
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
        nozzle_id = self.options["nozzle_id"]

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
            name="data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop",
            units="Pa",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            name="diffuser_exit_total_pressure",
            units="Pa",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            name="heat_exchanger_exit_temperature",
            units="K",
            val=np.nan,
        )
        self.add_input(
            "nozzle_air_density",
            units="kg/m**3",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "exterior_temperature",
            val=np.nan,
            units="K",
            shape=number_of_points,
        )
        self.add_input(
            "nozzle_air_specific_heat_capacity",
            units="J/kg/K",
            val=np.nan,
            shape=number_of_points,
        )
        self.add_input(
            "true_airspeed",
            units="m/s",
            val=np.nan,
            shape=number_of_points,
        )

        self.add_output("exit_pressure", val=1e5, units="Pa", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]
        self.declare_partials(
            of="*",
            wrt=[
                "exterior_temperature",
                "nozzle_air_density",
                "nozzle_air_specific_heat_capacity",
                "true_airspeed",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="heat_exchanger_exit_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="diffuser_exit_total_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0,
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + heat_exchanger_id
                + ":air_pressure_drop",
                "data:propulsion:he_power_train:PEMFC_stack_bop:"
                + pemfc_stack_bop_id
                + ":"
                + nozzle_id
                + ":air_pressure_drop",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pemfc_stack_bop_id = self.options["pemfc_stack_bop_id"]
        nozzle_id = self.options["nozzle_id"]
        heat_exchanger_id = self.options["connected_heat_exchanger_id"]

        total_pressure = inputs["diffuser_exit_total_pressure"]
        pressure_drop_heat_exchanger = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + heat_exchanger_id
            + ":air_pressure_drop"
        ]
        pressure_drop_nozzle = inputs[
            "data:propulsion:he_power_train:PEMFC_stack_bop:"
            + pemfc_stack_bop_id
            + ":"
            + nozzle_id
            + ":air_pressure_drop"
        ]

        inlet_temperature = inputs["heat_exchanger_exit_temperature"]
        exterior_temperature = inputs["exterior_temperature"]
        air_density = inputs["nozzle_air_density"]
        air_specific_heat_capacity = inputs["nozzle_air_specific_heat_capacity"]
        true_airspeed = inputs["true_airspeed"]

        outputs["exit_pressure"] = (
            total_pressure
            - pressure_drop_heat_exchanger
            - pressure_drop_nozzle
            + (
                air_specific_heat_capacity * (inlet_temperature - exterior_temperature)
                - 0.5 * true_airspeed**2.0
            )
            * air_density
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inlet_temperature = inputs["heat_exchanger_exit_temperature"]
        exterior_temperature = inputs["exterior_temperature"]
        air_density = inputs["nozzle_air_density"]
        air_specific_heat_capacity = inputs["nozzle_air_specific_heat_capacity"]
        true_airspeed = inputs["true_airspeed"]

        partials["exit_pressure", "nozzle_air_density"] = (
            air_specific_heat_capacity * (inlet_temperature - exterior_temperature)
            - 0.5 * true_airspeed**2.0
        )

        partials["exit_pressure", "heat_exchanger_exit_temperature"] = (
            air_specific_heat_capacity * air_density
        )

        partials["exit_pressure", "exterior_temperature"] = (
            -air_specific_heat_capacity * air_density
        )

        partials["exit_pressure", "nozzle_air_specific_heat_capacity"] = (
            inlet_temperature - exterior_temperature
        ) * air_density

        partials["exit_pressure", "true_airspeed"] = -air_density * true_airspeed

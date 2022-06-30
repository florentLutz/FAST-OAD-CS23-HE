# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesInducedSpeedConvergence(om.ImplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("elements_number", default=7, types=int)

    def setup(self):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]

        self.add_input(
            "thrust_tile_adt",
            units="N",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), np.nan),
        )
        self.add_input(
            "thrust_tile",
            units="N",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), np.nan),
        )
        self.add_input(
            "torque_tile_adt",
            units="N*m",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), np.nan),
        )
        self.add_input(
            "torque_tile",
            units="N*m",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), np.nan),
        )

        self.add_output(
            "v_i",
            val=[[1.9856], [5.5971], [7.9175]],
            shape=(elements_number, number_of_points),
            units="m/s",
        )
        self.add_output(
            name="v_t",
            val=[[6.0629], [9.6087], [9.4536]],
            shape=(elements_number, number_of_points),
            units="m/s",
        )

        self.declare_partials(of="*", wrt="*", method="fd", step=1e-2)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        residuals["v_i"] = (inputs["thrust_tile"] - inputs["thrust_tile_adt"]) / inputs[
            "thrust_tile_adt"
        ]
        residuals["v_t"] = (inputs["torque_tile"] - inputs["torque_tile_adt"]) / inputs[
            "torque_tile_adt"
        ]

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBladesSpeeds(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("elements_number", default=7, types=int)
        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "v_i",
            val=np.nan,
            shape=(elements_number, number_of_points),
            units="m/s",
        )
        self.add_input(
            "v_t",
            val=np.nan,
            shape=(elements_number, number_of_points),
            units="m/s",
        )
        self.add_input(
            "propeller_rpm",
            val=np.nan,
            shape=number_of_points,
            units="rad/s",
        )
        self.add_input(
            "true_airspeed",
            val=np.nan,
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            val=np.nan,
            shape=elements_number,
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_sweep",
            val=np.nan,
            shape=elements_number,
            units="rad",
        )

        self.add_output(
            "tot_v_axial",
            val=np.ones((elements_number, number_of_points)),
            units="m/s",
        )
        self.add_output(
            "tot_v_tangential",
            val=np.ones((elements_number, number_of_points)),
            units="m/s",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        elements_number = self.options["elements_number"]
        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]

        v_i = inputs["v_i"]
        v_t = inputs["v_t"]

        omega_tile = np.tile(inputs["propeller_rpm"], (elements_number, 1))
        v_inf_tile = np.tile(inputs["true_airspeed"], (elements_number, 1))
        r_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"],
            (number_of_points, 1),
        ).transpose()
        sweep_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_sweep"],
            (number_of_points, 1),
        ).transpose()

        v_ax = v_inf_tile + v_i
        v_tan = (omega_tile * r_tile - v_t) * np.cos(sweep_tile)

        outputs["tot_v_axial"] = v_ax
        outputs["tot_v_tangential"] = v_tan

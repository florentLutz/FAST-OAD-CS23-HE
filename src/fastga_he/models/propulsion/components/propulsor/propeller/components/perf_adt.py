# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from stdatm import Atmosphere


class PerformancesADT(om.ExplicitComponent):
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
            "altitude",
            val=np.nan,
            shape=number_of_points,
            units="m",
        )
        self.add_input(
            "tot_v_axial",
            val=np.full((elements_number, number_of_points), np.nan),
            units="m/s",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            val=np.nan,
            shape=elements_number,
            units="m",
        )
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
            "tip_loss",
            val=np.full((elements_number, number_of_points), np.nan),
        )

        self.add_output(
            "thrust_tile_adt",
            units="N",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), 10.0),
        )
        self.add_output(
            "torque_tile_adt",
            units="N*m",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), 1.0),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        v_ax = inputs["tot_v_axial"]
        v_i = inputs["v_i"]
        v_t = inputs["v_t"]

        tip_loss = inputs["tip_loss"]

        r_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"],
            (number_of_points, 1),
        ).transpose()

        atm = Atmosphere(inputs["altitude"], altitude_in_feet=False)
        density_tile = np.tile(atm.density, (elements_number, 1))

        thrust_tile = 4.0 * np.pi * density_tile * r_tile * v_ax * v_i * tip_loss
        torque_tile = 4.0 * np.pi * density_tile * r_tile ** 2.0 * v_ax * v_t * tip_loss

        outputs["thrust_tile_adt"] = thrust_tile
        outputs["torque_tile_adt"] = torque_tile


class PerformancesADTToLoop(om.ExplicitComponent):
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
            "altitude",
            val=np.nan,
            shape=number_of_points,
            units="m",
        )
        self.add_input(
            "tot_v_axial",
            val=np.full((elements_number, number_of_points), np.nan),
            units="m/s",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            val=np.nan,
            shape=elements_number,
            units="m",
        )
        self.add_input(
            "tip_loss",
            val=np.full((elements_number, number_of_points), np.nan),
        )
        self.add_input(
            "thrust_tile",
            units="N",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), 10.0),
        )
        self.add_input(
            "torque_tile",
            units="N*m",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), 1.0),
        )

        self.add_output(
            "v_i",
            val=[[1.9856], [5.5971], [7.9175]],
            shape=(elements_number, number_of_points),
            units="m/s",
        )
        self.add_output(
            "v_t",
            val=[[6.0629], [9.6087], [9.4536]],
            shape=(elements_number, number_of_points),
            units="m/s",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        v_ax = inputs["tot_v_axial"]

        tip_loss = inputs["tip_loss"]

        thrust_tile = inputs["thrust_tile"]
        torque_tile = inputs["torque_tile"]

        r_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"],
            (number_of_points, 1),
        ).transpose()

        atm = Atmosphere(inputs["altitude"], altitude_in_feet=False)
        density_tile = np.tile(atm.density, (elements_number, 1))

        v_i = thrust_tile / (4.0 * np.pi * density_tile * r_tile * v_ax * tip_loss)
        v_t = torque_tile / (4.0 * np.pi * density_tile * r_tile ** 2.0 * v_ax * tip_loss)

        outputs["v_i"] = v_i
        outputs["v_t"] = v_t

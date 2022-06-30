# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from stdatm import Atmosphere


class PerformancesBEMT(om.ExplicitComponent):
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
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blades_number",
            val=np.nan,
        )
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
            "tot_v_tangential",
            val=np.full((elements_number, number_of_points), np.nan),
            units="m/s",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_chord",
            shape=elements_number,
            units="m",
            val=np.full(elements_number, np.nan),
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            val=np.nan,
            shape=elements_number,
            units="m",
        )
        self.add_input(
            "blades_cl",
            val=np.full((elements_number, number_of_points), np.nan),
        )
        self.add_input(
            "blades_cd",
            val=np.full((elements_number, number_of_points), np.nan),
        )

        self.add_output(
            "thrust_tile",
            units="N",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), 10.0),
        )
        self.add_output(
            "torque_tile",
            units="N*m",
            shape=(elements_number, number_of_points),
            val=np.full((elements_number, number_of_points), 1.0),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        blades_number = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blades_number"
        ]
        v_ax = inputs["tot_v_axial"]
        v_tan = inputs["tot_v_tangential"]

        chord_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_chord"],
            (number_of_points, 1),
        ).transpose()
        r_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"],
            (number_of_points, 1),
        ).transpose()

        phi = np.arctan2(v_ax, v_tan)
        tot_rel_speed = np.sqrt(v_ax ** 2.0 + v_tan ** 2.0)

        atm = Atmosphere(inputs["altitude"], altitude_in_feet=False)
        sos_tile = np.tile(atm.speed_of_sound, (elements_number, 1))
        density_tile = np.tile(atm.density, (elements_number, 1))

        mach_tile = tot_rel_speed / sos_tile
        coeff_cl = np.where(
            mach_tile < 1.0, np.sqrt(1.0 - mach_tile ** 2.0), np.sqrt(mach_tile ** 2.0 - 1)
        )
        coeff_cd = np.where(mach_tile < 1.0, 1.0, np.sqrt(mach_tile ** 2.0 - 1))

        actual_cl = inputs["blades_cl"] / coeff_cl
        actual_cd = inputs["blades_cd"] / coeff_cd

        thrust_tile = (
            0.5
            * density_tile
            * blades_number
            * chord_tile
            * tot_rel_speed ** 2.0
            * (actual_cl * np.cos(phi) - actual_cd * np.sin(phi))
        )
        torque_tile = (
            0.5
            * density_tile
            * blades_number
            * chord_tile
            * r_tile
            * tot_rel_speed ** 2.0
            * (actual_cl * np.sin(phi) + actual_cd * np.cos(phi))
        )

        outputs["thrust_tile"] = thrust_tile
        outputs["torque_tile"] = torque_tile

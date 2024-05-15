# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesVoltageOutTargetMission(om.ExplicitComponent):
    """
    Component which takes the desired voltage output target for the turbo generator operation
    from the data and gives it the right format for the mission. It was deemed best to put it
    this way rather than the original way to simplify the construction of the power train file.

    The input voltage target can either be a float (then during the whole mission the
    voltage is going to be the same) or an array of number of points elements for the
    individual control of each point.
    """

    def initialize(self):

        self.options.declare(
            name="turbo_generator_id",
            default=None,
            desc="Identifier of the turbo generator",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":voltage_out_target_mission",
            val=np.nan,
            units="V",
            desc="Target output voltage of the turbo generator for the points",
            shape_by_conn=True,
        )

        self.add_output("voltage_out_target", units="V", val=850.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        v_out_tgt_mission = inputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":voltage_out_target_mission"
        ]

        if len(v_out_tgt_mission) == 1:
            outputs["voltage_out_target"] = np.full(number_of_points, v_out_tgt_mission)

        elif len(v_out_tgt_mission) == number_of_points:
            outputs["voltage_out_target"] = v_out_tgt_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":voltage_out_target_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turbo_generator_id = self.options["turbo_generator_id"]
        number_of_points = self.options["number_of_points"]

        v_out_tgt_mission = inputs[
            "data:propulsion:he_power_train:turbo_generator:"
            + turbo_generator_id
            + ":voltage_out_target_mission"
        ]

        if len(v_out_tgt_mission) == 1:
            partials[
                "voltage_out_target",
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":voltage_out_target_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "voltage_out_target",
                "data:propulsion:he_power_train:turbo_generator:"
                + turbo_generator_id
                + ":voltage_out_target_mission",
            ] = np.eye(number_of_points)

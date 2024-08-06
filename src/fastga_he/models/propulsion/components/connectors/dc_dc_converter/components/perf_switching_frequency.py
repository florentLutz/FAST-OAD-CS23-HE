# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesSwitchingFrequencyMission(om.ExplicitComponent):
    """
    Component which takes the desired switching frequency for converter operation from the data
    and gives it the right format for the mission. It was deemed best to put it this way rather
    than the original way to simplify the construction of the power train file.

    The input switching frequency can either be a float (then during the whole mission the
    frequency is going to be the same) or an array of number of points elements for the
    individual control of each point.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_dc_converter_id",
            default=None,
            desc="Identifier of the DC/DC converter",
            allow_none=False,
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_mission",
            val=np.nan,
            units="Hz",
            desc="Switching frequency of the DC/DC converter for the points",
            shape_by_conn=True,
        )

        self.add_output("switching_frequency", units="Hz", val=15.0e3, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        f_switch_mission = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_mission"
        ]

        if len(f_switch_mission) == 1:
            outputs["switching_frequency"] = np.full(number_of_points, f_switch_mission)

        elif len(f_switch_mission) == number_of_points:
            outputs["switching_frequency"] = f_switch_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        f_switch_mission = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":switching_frequency_mission"
        ]

        if len(f_switch_mission) == 1:
            partials[
                "switching_frequency",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "switching_frequency",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":switching_frequency_mission",
            ] = np.eye(number_of_points)

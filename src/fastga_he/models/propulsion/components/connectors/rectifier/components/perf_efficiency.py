# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.exceptions import ControlParameterInconsistentShapeError


class PerformancesEfficiencyMission(om.ExplicitComponent):
    """
    Component which takes the efficiency of the rectifier from the data and gives it the right
    format for the mission. It is a temporary solution as it will later be computed based on
    switching losses, conduction losses, ...
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":efficiency",
            val=np.nan,
            desc="Efficiency of the rectifier for the points",
            shape_by_conn=True,
        )

        self.add_output("efficiency", val=1.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        eta = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":efficiency"]

        if len(eta) == 1:
            outputs["efficiency"] = np.full(number_of_points, eta)

        elif len(eta) == number_of_points:
            outputs["efficiency"] = eta

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:rectifier:"
                + rectifier_id
                + ":efficiency"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        eta = inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":efficiency"]

        if len(eta) == 1:
            partials[
                "efficiency",
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":efficiency",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "efficiency",
                "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":efficiency",
            ] = np.eye(number_of_points)

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om



class PerformancesEfficiency(om.ExplicitComponent):
    """
    Component of the power at the output side of the converter.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("dc_voltage_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("power_out", val=1.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        eta_mission = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":efficiency_mission"
        ]

        if len(eta_mission) == 1:
            outputs["efficiency"] = np.full(number_of_points, eta_mission)

        elif len(eta_mission) == number_of_points:
            outputs["efficiency"] = eta_mission

        else:
            raise ControlParameterInconsistentShapeError(
                "The shape of input "
                + "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":efficiency_mission"
                + " should be 1 or equal to the number of points"
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        dc_dc_converter_id = self.options["dc_dc_converter_id"]
        number_of_points = self.options["number_of_points"]

        eta_mission = inputs[
            "data:propulsion:he_power_train:DC_DC_converter:"
            + dc_dc_converter_id
            + ":efficiency_mission"
        ]

        if len(eta_mission) == 1:
            partials[
                "efficiency",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":efficiency_mission",
            ] = np.full(number_of_points, 1.0)

        else:
            partials[
                "efficiency",
                "data:propulsion:he_power_train:DC_DC_converter:"
                + dc_dc_converter_id
                + ":efficiency_mission",
            ] = np.eye(number_of_points)

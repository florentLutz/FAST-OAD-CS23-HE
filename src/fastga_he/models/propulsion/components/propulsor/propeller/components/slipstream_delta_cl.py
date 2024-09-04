# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerDeltaCl(om.ExplicitComponent):
    """
    Compute the increase in lift coefficient due to the blowing of the wing. Considers the 2D
    effect and a blown area ratio to transpose to 3D.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "delta_Cl_2D",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in the section lift downstream of the propeller",
        )
        self.add_input(
            "delta_Cl_2D_AOA_0",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in the section lift downstream of the propeller for a nil AOA",
        )
        self.add_input(
            "blown_area_ratio",
            val=np.nan,
            shape=number_of_points,
            desc="Portion of the wing blown by the propeller",
        )

        self.add_output(
            "delta_Cl",
            val=0.01,
            shape=number_of_points,
            desc="Increase in the lift coefficient downstream of the propeller",
        )
        self.add_output(
            "delta_Cl_AOA_0",
            val=0.01,
            shape=number_of_points,
            desc="Increase in the lift coefficient downstream of the propeller for a nil AOA",
        )

        self.declare_partials(
            of="delta_Cl",
            wrt=["blown_area_ratio", "delta_Cl_2D"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="delta_Cl_AOA_0",
            wrt=["blown_area_ratio", "delta_Cl_2D_AOA_0"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["delta_Cl"] = inputs["delta_Cl_2D"] * inputs["blown_area_ratio"]
        outputs["delta_Cl_AOA_0"] = inputs["delta_Cl_2D_AOA_0"] * inputs["blown_area_ratio"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["delta_Cl", "delta_Cl_2D"] = inputs["blown_area_ratio"]
        partials["delta_Cl", "blown_area_ratio"] = inputs["delta_Cl_2D"]

        partials["delta_Cl_AOA_0", "delta_Cl_2D_AOA_0"] = inputs["blown_area_ratio"]
        partials["delta_Cl_AOA_0", "blown_area_ratio"] = inputs["delta_Cl_2D_AOA_0"]

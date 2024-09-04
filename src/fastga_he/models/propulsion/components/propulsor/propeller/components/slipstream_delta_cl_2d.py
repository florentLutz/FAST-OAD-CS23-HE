# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerDeltaCl2D(om.ExplicitComponent):
    """
    From the increase ratio and the unblown lift coefficient, computes the delta Cl caused by
    propeller slipstream.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "unblown_section_lift",
            val=np.nan,
            desc="Value of the unblown lift downstream of the propeller",
            shape=number_of_points,
        )
        self.add_input(
            "unblown_section_lift_AOA_0",
            val=np.nan,
            desc="Value of the unblown lift downstream of the propeller for a nil AOA",
            shape=number_of_points,
        )
        self.add_input(
            "lift_increase_ratio",
            val=np.nan,
            shape=number_of_points,
            desc="Increase in lift due to the slipstream effect behind the propeller expressed as"
            " a ratio of the clean lift",
        )

        self.add_output(
            "delta_Cl_2D",
            val=0.6,
            desc="Increase in the section lift downstream of the propeller",
            shape=number_of_points,
        )
        self.add_output(
            "delta_Cl_2D_AOA_0",
            val=0.3,
            desc="Increase in the section lift downstream of the propeller for a nil AOA",
            shape=number_of_points,
        )

        self.declare_partials(
            of="delta_Cl_2D",
            wrt=["unblown_section_lift", "lift_increase_ratio"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="delta_Cl_2D_AOA_0",
            wrt=["unblown_section_lift_AOA_0", "lift_increase_ratio"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["delta_Cl_2D"] = inputs["unblown_section_lift"] * inputs["lift_increase_ratio"]
        outputs["delta_Cl_2D_AOA_0"] = (
            inputs["unblown_section_lift_AOA_0"] * inputs["lift_increase_ratio"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["delta_Cl_2D", "unblown_section_lift"] = inputs["lift_increase_ratio"]
        partials["delta_Cl_2D", "lift_increase_ratio"] = inputs["unblown_section_lift"]

        partials["delta_Cl_2D_AOA_0", "unblown_section_lift_AOA_0"] = inputs["lift_increase_ratio"]
        partials["delta_Cl_2D_AOA_0", "lift_increase_ratio"] = inputs["unblown_section_lift_AOA_0"]

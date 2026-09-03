# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalCalculateNPAXDesign(om.ExplicitComponent):
    """
    Computation of the number of passenger carried per flight mission.
    """

    def setup(self):
        self.add_input(
            "settings:weight:aircraft:payload:design_mass_per_passenger",
            val=np.nan,
            units="kg",
            desc="design payload mass carried by passenger",
        )
        self.add_input(
            "settings:weight:aircraft:payload:design_luggage_per_passenger",
            val=8.0,
            units="kg",
            desc="design luggage mass carried by passenger",
        )
        self.add_input(
            "settings:mission:number_of_pilot",
            val=2.0,
            desc="Number of pilots for the sizing mission",
        )
        self.add_input(
            "data:weight:aircraft:payload",
            units="kg",
            val=np.nan,
        )

        self.add_output(
            "data:TLAR:NPAX_design",
            val=4.0,
            desc="Number of passengers for the sizing mission",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        design_mass_per_passenger = inputs[
            "settings:weight:aircraft:payload:design_mass_per_passenger"
        ]
        design_luggage_per_passenger = inputs[
            "settings:weight:aircraft:payload:design_luggage_per_passenger"
        ]
        payload = inputs["data:weight:aircraft:payload"]
        number_of_pilot = inputs["settings:mission:number_of_pilot"]

        outputs["data:TLAR:NPAX_design"] = (
            payload / (design_mass_per_passenger + design_luggage_per_passenger) - number_of_pilot
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        design_mass_per_passenger = inputs[
            "settings:weight:aircraft:payload:design_mass_per_passenger"
        ]
        design_luggage_per_passenger = inputs[
            "settings:weight:aircraft:payload:design_luggage_per_passenger"
        ]
        payload = inputs["data:weight:aircraft:payload"]

        partials[
            "data:TLAR:NPAX_design", "settings:weight:aircraft:payload:design_mass_per_passenger"
        ] = -payload / (design_mass_per_passenger + design_luggage_per_passenger) ** 2.0

        partials[
            "data:TLAR:NPAX_design", "settings:weight:aircraft:payload:design_luggage_per_passenger"
        ] = -payload / (design_mass_per_passenger + design_luggage_per_passenger) ** 2.0

        partials["data:TLAR:NPAX_design", "data:weight:aircraft:payload"] = 1.0 / (
            design_mass_per_passenger + design_luggage_per_passenger
        )

        partials["data:TLAR:NPAX_design", "settings:mission:number_of_pilot"] = -1.0

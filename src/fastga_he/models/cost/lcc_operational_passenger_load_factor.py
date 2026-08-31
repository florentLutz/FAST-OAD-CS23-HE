# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalPassengerLoadFactor(om.ExplicitComponent):
    """
    Computation of the passenger load factor of the aircraft. This is simplified as the ratio
    between the number of seats for the sizing mission and the maximum number of seats.
    """

    def setup(self):
        self.add_input(
            "data:TLAR:NPAX_design",
            val=np.nan,
            desc="Number of passengers for the sizing mission",
        )
        self.add_input(
            "data:geometry:cabin:seats:passenger:NPAX_max",
            val=np.nan,
            desc="Maximum number of passengers that can be seated in the aircraft",
        )

        self.add_output("data:cost:operation:passenger_load_factor", val=0.75)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:passenger_load_factor"] = (
            inputs["data:TLAR:NPAX_design"] / inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:operation:passenger_load_factor", "data:TLAR:NPAX_design"] = (
            1.0 / inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        )

        partials[
            "data:cost:operation:passenger_load_factor",
            "data:geometry:cabin:seats:passenger:NPAX_max",
        ] = (
            -inputs["data:TLAR:NPAX_design"]
            / inputs["data:geometry:cabin:seats:passenger:NPAX_max"] ** 2.0
        )

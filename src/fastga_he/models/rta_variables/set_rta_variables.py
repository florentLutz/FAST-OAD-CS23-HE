# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterOpenMDAOSystem("fastga_he.rta_variables.set_rta_variable")
class SetRTAVariable(om.ExplicitComponent):
    """Some variables with standard values are required in HE as input,
    but they are not required as input in RTA, or some values are to complex to be computed"""

    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)  # not used

        # standard value
        self.add_output("data:mission:sizing:takeoff:energy", units="W*h")
        self.add_output("data:mission:sizing:initial_climb:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:reserve:duration", units="s")
        self.add_output("data:geometry:cabin:seats:pilot:length", units="m")
        self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
        self.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="1/rad")

        # performance required as input
        self.add_output("data:mission:sizing:main_route:climb:climb_rate:cruise_level", units="m/s")
        self.add_output("data:mission:sizing:main_route:climb:climb_rate:sea_level", units="m/s")
        self.add_output("data:mission:sizing:main_route:descent:descent_rate", units="m/s")

        # elevator info required as input
        self.add_output("data:geometry:horizontal_tail:elevator_chord_ratio")
        self.add_output("data:mission:sizing:landing:elevator_angle", units="rad")

        # more airfoil input required: dy, maximum camber and its position, position of maximum thickness
        self.add_output("data:aerodynamics:wing:cruise:CM0_clean")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:mission:sizing:takeoff:energy"] = 0

        outputs["data:mission:sizing:initial_climb:energy"] = 0

        outputs["data:mission:sizing:main_route:reserve:duration"] = 3500.0

        outputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"] = 4.0

        outputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"] = 9.0

        outputs["data:mission:sizing:main_route:descent:descent_rate"] = -7.62 * 2.0 / 3.0

        outputs["data:geometry:horizontal_tail:elevator_chord_ratio"] = 0.384

        outputs[
            "data:mission:sizing:landing:elevator_angle"
        ] = -0.6363323129985824  # -0.6363323129985824

        outputs["data:geometry:cabin:seats:pilot:length"] = 1.05

        outputs["data:aerodynamics:horizontal_tail:cruise:CL0"] = -0.0068437669175491515

        outputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"] = 6.28

        outputs["data:aerodynamics:wing:cruise:CM0_clean"] = -0.02413516654351498

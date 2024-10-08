# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class SizingEnergy(om.ExplicitComponent):
    """Computes the fuel consumed during the whole sizing mission."""

    def setup(self):
        self.add_input("data:mission:sizing:main_route:climb:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:descent:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:descent:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:reserve:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:reserve:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:taxi_out:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:takeoff:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:initial_climb:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:energy", val=np.nan, units="W*h")
        self.add_input("data:mission:sizing:taxi_in:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_in:energy", val=np.nan, units="W*h")

        self.add_output("data:mission:sizing:fuel", val=250, units="kg")
        self.add_output("data:mission:sizing:energy", val=200e3, units="W*h")

        self.declare_partials(of="data:mission:sizing:fuel", wrt="*:fuel", method="exact", val=1.0)
        self.declare_partials(
            of="data:mission:sizing:energy", wrt="*:energy", method="exact", val=1.0
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:mission:sizing:fuel"] = (
            inputs["data:mission:sizing:main_route:climb:fuel"]
            + inputs["data:mission:sizing:main_route:cruise:fuel"]
            + inputs["data:mission:sizing:main_route:descent:fuel"]
            + inputs["data:mission:sizing:main_route:reserve:fuel"]
            + inputs["data:mission:sizing:taxi_out:fuel"]
            + inputs["data:mission:sizing:taxi_in:fuel"]
            + inputs["data:mission:sizing:takeoff:fuel"]
            + inputs["data:mission:sizing:initial_climb:fuel"]
        )

        outputs["data:mission:sizing:energy"] = (
            inputs["data:mission:sizing:main_route:climb:energy"]
            + inputs["data:mission:sizing:main_route:cruise:energy"]
            + inputs["data:mission:sizing:main_route:descent:energy"]
            + inputs["data:mission:sizing:main_route:reserve:energy"]
            + inputs["data:mission:sizing:taxi_out:energy"]
            + inputs["data:mission:sizing:taxi_in:energy"]
            + inputs["data:mission:sizing:takeoff:energy"]
            + inputs["data:mission:sizing:initial_climb:energy"]
        )

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
        self.add_output(
            "data:mission:sizing:main_route:fuel",
            val=250,
            units="kg",
            desc="Fuel burn on the mission, excluding reserves",
        )
        self.add_output("data:mission:sizing:energy", val=200e3, units="W*h")
        self.add_output(
            "data:mission:sizing:main_route:energy",
            val=200e3,
            units="W*h",
            desc="Energy required for the mission, excluding reserves",
        )

    def setup_partials(self):
        self.declare_partials(of="data:mission:sizing:fuel", wrt="*:fuel", method="exact", val=1.0)
        self.declare_partials(
            of="data:mission:sizing:main_route:fuel",
            wrt=[
                "data:mission:sizing:main_route:climb:fuel",
                "data:mission:sizing:main_route:cruise:fuel",
                "data:mission:sizing:main_route:descent:fuel",
                "data:mission:sizing:taxi_out:fuel",
                "data:mission:sizing:taxi_in:fuel",
                "data:mission:sizing:takeoff:fuel",
                "data:mission:sizing:initial_climb:fuel",
            ],
            method="exact",
            val=1.0,
        )
        self.declare_partials(
            of="data:mission:sizing:energy", wrt="*:energy", method="exact", val=1.0
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:energy",
            wrt=[
                "data:mission:sizing:main_route:climb:energy",
                "data:mission:sizing:main_route:cruise:energy",
                "data:mission:sizing:main_route:descent:energy",
                "data:mission:sizing:taxi_out:energy",
                "data:mission:sizing:taxi_in:energy",
                "data:mission:sizing:takeoff:energy",
                "data:mission:sizing:initial_climb:energy",
            ],
            method="exact",
            val=1.0,
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
        outputs["data:mission:sizing:main_route:fuel"] = (
            inputs["data:mission:sizing:main_route:climb:fuel"]
            + inputs["data:mission:sizing:main_route:cruise:fuel"]
            + inputs["data:mission:sizing:main_route:descent:fuel"]
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
        outputs["data:mission:sizing:main_route:energy"] = (
            inputs["data:mission:sizing:main_route:climb:energy"]
            + inputs["data:mission:sizing:main_route:cruise:energy"]
            + inputs["data:mission:sizing:main_route:descent:energy"]
            + inputs["data:mission:sizing:taxi_out:energy"]
            + inputs["data:mission:sizing:taxi_in:energy"]
            + inputs["data:mission:sizing:takeoff:energy"]
            + inputs["data:mission:sizing:initial_climb:energy"]
        )

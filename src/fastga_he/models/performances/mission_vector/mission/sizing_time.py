# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class SizingDuration(om.ExplicitComponent):
    """Computes the duration whole sizing mission."""

    def setup(self):
        self.add_input("data:mission:sizing:main_route:climb:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:cruise:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:descent:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:reserve:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_out:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_in:duration", val=np.nan, units="s")

        self.add_output("data:mission:sizing:duration", val=3600.0, units="s")
        self.add_output("data:mission:sizing:main_route:duration", val=3600.0, units="s")

    def setup_partials(self):
        self.declare_partials(
            of="data:mission:sizing:duration", wrt="*:duration", method="exact", val=1.0
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:duration",
            wrt=[
                "data:mission:sizing:main_route:climb:duration",
                "data:mission:sizing:main_route:cruise:duration",
                "data:mission:sizing:main_route:descent:duration",
                "data:mission:sizing:taxi_out:duration",
                "data:mission:sizing:taxi_in:duration",
            ],
            method="exact",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:mission:sizing:duration"] = (
            inputs["data:mission:sizing:main_route:climb:duration"]
            + inputs["data:mission:sizing:main_route:cruise:duration"]
            + inputs["data:mission:sizing:main_route:descent:duration"]
            + inputs["data:mission:sizing:main_route:reserve:duration"]
            + inputs["data:mission:sizing:taxi_out:duration"]
            + inputs["data:mission:sizing:taxi_in:duration"]
        )

        outputs["data:mission:sizing:main_route:duration"] = (
            inputs["data:mission:sizing:main_route:climb:duration"]
            + inputs["data:mission:sizing:main_route:cruise:duration"]
            + inputs["data:mission:sizing:main_route:descent:duration"]
            + inputs["data:mission:sizing:taxi_out:duration"]
            + inputs["data:mission:sizing:taxi_in:duration"]
        )

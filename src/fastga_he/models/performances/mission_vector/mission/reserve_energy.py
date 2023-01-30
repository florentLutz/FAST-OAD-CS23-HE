# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

from deprecated import deprecated

import numpy as np
import openmdao.api as om


@deprecated(
    version="0.0.1",
    reason="This method is no more the way we compute reserve, it is now computed as a proper "
    "flight phase",
)
class ReserveEnergy(om.ExplicitComponent):
    """Computes the fuel consumed during the reserve phase."""

    def setup(self):

        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:energy", np.nan, units="W*h")
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:reserve:duration", np.nan, units="s")

        self.add_input("settings:mission:sizing:main_route:reserve:k_factor", val=1.0)

        self.add_output("data:mission:sizing:main_route:reserve:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:reserve:energy", units="W*h")

    def setup_partials(self):
        self.declare_partials(
            of="data:mission:sizing:main_route:reserve:fuel",
            wrt=[
                "data:mission:sizing:main_route:cruise:fuel",
                "data:mission:sizing:main_route:cruise:duration",
                "data:mission:sizing:main_route:reserve:duration",
                "settings:mission:sizing:main_route:reserve:k_factor",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:reserve:energy",
            wrt=[
                "data:mission:sizing:main_route:cruise:energy",
                "data:mission:sizing:main_route:cruise:duration",
                "data:mission:sizing:main_route:reserve:duration",
                "settings:mission:sizing:main_route:reserve:k_factor",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_reserve = (
            inputs["data:mission:sizing:main_route:cruise:fuel"]
            * inputs["data:mission:sizing:main_route:reserve:duration"]
            * inputs["settings:mission:sizing:main_route:reserve:k_factor"]
            / max(
                1e-6, inputs["data:mission:sizing:main_route:cruise:duration"]
            )  # avoid 0 division
        )
        energy_reserve = (
            inputs["data:mission:sizing:main_route:cruise:energy"]
            * inputs["data:mission:sizing:main_route:reserve:duration"]
            * inputs["settings:mission:sizing:main_route:reserve:k_factor"]
            / max(
                1e-6, inputs["data:mission:sizing:main_route:cruise:duration"]
            )  # avoid 0 division
        )
        outputs["data:mission:sizing:main_route:reserve:fuel"] = m_reserve
        outputs["data:mission:sizing:main_route:reserve:energy"] = energy_reserve

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cruise_time = inputs["data:mission:sizing:main_route:cruise:duration"]
        rsv_time = inputs["data:mission:sizing:main_route:reserve:duration"]
        cruise_fuel = inputs["data:mission:sizing:main_route:cruise:fuel"]
        cruise_energy = inputs["data:mission:sizing:main_route:cruise:energy"]
        k_factor = inputs["settings:mission:sizing:main_route:reserve:k_factor"]

        partials[
            "data:mission:sizing:main_route:reserve:fuel",
            "data:mission:sizing:main_route:cruise:fuel",
        ] = (
            rsv_time * k_factor / cruise_time
        )
        partials[
            "data:mission:sizing:main_route:reserve:energy",
            "data:mission:sizing:main_route:cruise:energy",
        ] = (
            rsv_time * k_factor / cruise_time
        )
        partials[
            "data:mission:sizing:main_route:reserve:fuel",
            "data:mission:sizing:main_route:reserve:duration",
        ] = (
            cruise_fuel * k_factor / cruise_time
        )
        partials[
            "data:mission:sizing:main_route:reserve:energy",
            "data:mission:sizing:main_route:reserve:duration",
        ] = (
            cruise_energy * k_factor / cruise_time
        )
        partials[
            "data:mission:sizing:main_route:reserve:fuel",
            "data:mission:sizing:main_route:cruise:duration",
        ] = (
            rsv_time * cruise_fuel * k_factor / cruise_time ** 2
        )
        partials[
            "data:mission:sizing:main_route:reserve:energy",
            "data:mission:sizing:main_route:cruise:duration",
        ] = (
            rsv_time * cruise_energy * k_factor / cruise_time ** 2
        )

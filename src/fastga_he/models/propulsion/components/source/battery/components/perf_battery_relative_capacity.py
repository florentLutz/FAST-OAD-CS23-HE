# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesRelativeCapacity(om.ExplicitComponent):
    """
    Computation of the relative capacity of each module. As described in :cite:`vratny:2013` the
    capacity used for the computation of the change in SOC depends on the current drawn. A simple
    polynomial interpolation is implemented based on the value from :cite:`samsung:2015` .
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.poly = None
        self.der_poly = None

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "reference_curve_current",
            default=[680, 3400, 6800, 8000],
            desc="Data for the relative capacity curve of the reference cell, in mA",
        )
        self.options.declare(
            "reference_curve_relative_capacity",
            default=[1, 0.97, 0.95, 0.92],
            desc="Data for the relative capacity curve of the reference cell",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("current_one_module", units="mA", val=np.full(number_of_points, np.nan))

        self.add_output("relative_capacity", val=np.full(number_of_points, 1.0))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.poly = np.polyfit(
            self.options["reference_curve_current"],
            self.options["reference_curve_relative_capacity"],
            3,
        )
        self.der_poly = np.polyder(self.poly)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        current = inputs["current_one_module"]
        relative_capacity = np.polyval(self.poly, current)

        # The capacity can't be > 1.0 even if the interpolation allows it for very small
        # current. Likewise, if the relative capacity is below 0.85 (somewhat arbitrary value,
        # is 0.92 for this particular cell), it means it is above its max discharge current,
        # which can happen in early loops, so we will clip it
        outputs["relative_capacity"] = np.clip(
            relative_capacity,
            np.full_like(relative_capacity, 0.85),
            np.ones_like(relative_capacity),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        current = inputs["current_one_module"]

        relative_capacity = np.polyval(self.poly, current)
        d_r_d_current = np.polyval(self.der_poly, current)
        d_r_d_current = np.where(
            relative_capacity < 1.0, d_r_d_current, np.zeros_like(d_r_d_current)
        )
        d_r_d_current = np.where(
            relative_capacity > 0.0, d_r_d_current, np.zeros_like(d_r_d_current)
        )

        partials["relative_capacity", "current_one_module"] = d_r_d_current

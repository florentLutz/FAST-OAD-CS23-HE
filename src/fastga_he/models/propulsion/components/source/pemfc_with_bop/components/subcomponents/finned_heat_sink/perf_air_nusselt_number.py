# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAirNusseltNumber(om.ExplicitComponent):
    """
    Computation of the air Nusselt number.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(name="reynolds_number", units="unitless", val=np.nan, shape=number_of_points)
        self.add_input("prandtl_number", units="unitless", val=np.nan, shape=number_of_points)

        self.add_output(name="nusselt_number", units="unitless", val=400.0)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        reynolds_number = inputs["reynolds_number"]
        prandtl_number = inputs["prandtl_number"]

        outputs["nusselt_number"] = np.max(
            (
                (0.5 * reynolds_number * prandtl_number) ** -3
                + (
                    0.664
                    * reynolds_number**0.5
                    * prandtl_number ** (1.0 / 3.0)
                    * np.sqrt(1.0 + 3.65 / reynolds_number**0.5)
                )
                ** -3
            )
            ** (-1.0 / 3.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        re = inputs["reynolds_number"]
        pr = inputs["prandtl_number"]

        nusselt_number = (
            (0.5 * re * pr) ** -3
            + (0.664 * re**0.5 * pr ** (1.0 / 3.0) * np.sqrt(1.0 + 3.65 / re**0.5)) ** -3
        ) ** (-1.0 / 3.0)

        max_nusselt_number = np.max(nusselt_number)

        a = 0.5 * re * pr
        b = 0.664 * re**0.5 * pr ** (1.0 / 3.0) * np.sqrt(1.0 + 3.65 / re**0.5)

        f = a ** (-3) + b ** (-3)

        da_dre = 0.5 * pr
        da_dpr = 0.5 * re

        db_dre = (
            0.664
            * pr ** (1.0 / 3.0)
            * (1.0 + 1.825 * re ** (-0.5))
            / (2.0 * re**0.5 * np.sqrt(1.0 + 3.65 * re ** (-0.5)))
        )
        db_dpr = b / (3.0 * pr)

        partials["nusselt_number", "reynolds_number"] = np.where(
            nusselt_number == max_nusselt_number,
            f ** (-4.0 / 3.0) * (a ** (-4) * da_dre + b ** (-4) * db_dre),
            1e-6,
        )

        partials["nusselt_number", "prandtl_number"] = np.where(
            nusselt_number == max_nusselt_number,
            f ** (-4.0 / 3.0) * (a ** (-4) * da_dpr + b ** (-4) * db_dpr),
            1e-6,
        )

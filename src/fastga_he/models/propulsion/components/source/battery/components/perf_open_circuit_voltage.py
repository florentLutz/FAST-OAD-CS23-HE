# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesOpenCircuitVoltage(om.ExplicitComponent):
    """
    Computation of the open circuit voltage of one module cell, takes into account the impact of
    the SOC on the performances. Does not account for temperature just yet, it seems however that
    the dependency on temperature is only visible at very low SOC :cite:`chin:2019`. The model
    put forward by :cite:`baccouche:2017` does not provide satisfactory results but the article
    suggest a simple polynomial fit could provide adequate results, which is what we took. See
    internal_resistance_new.py for how we obtain those values.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))

        self.add_output("open_circuit_voltage", units="V", val=np.full(number_of_points, 4.1))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )
        dod = 100.0 - soc

        ocv = (
            -9.65121262e-10 * dod ** 5.0
            + 1.81419058e-07 * dod ** 4.0
            - 1.11814100e-05 * dod ** 3.0
            + 2.26114438e-04 * dod ** 2.0
            - 8.54619953e-03 * dod
            + 4.12
        )
        outputs["open_circuit_voltage"] = ocv

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10 - 1e-3),
            np.full_like(inputs["state_of_charge"], 100 + 1e-3),
        )
        dod = 100.0 - soc

        partials["open_circuit_voltage", "state_of_charge"] = -(
            -5.0 * 9.65121262e-10 * dod ** 4.0
            + 4.0 * 1.81419058e-07 * dod ** 3.0
            - 3.0 * 1.11814100e-05 * dod ** 2.0
            + 2.0 * 2.26114438e-04 * dod
            - 8.54619953e-03
        )

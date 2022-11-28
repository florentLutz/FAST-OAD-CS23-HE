# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesOpenCircuitVoltage(om.ExplicitComponent):
    """
    Computation of the open circuit voltage of one module cell, takes into account the impact of
    the SOC on the performances. Does not account for temperature just yet. As a matter of fact,
    this component, an implementation of the model from :cite:`baccouche:2017`, will use only the
    coefficient at 25 degC.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))

        self.add_output("open_circuit_voltage", units="V", val=np.full(number_of_points, 4.1))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        soc = np.clip(
            inputs["state_of_charge"],
            np.full_like(inputs["state_of_charge"], 10),
            np.full_like(inputs["state_of_charge"], 100),
        )

        ocv = (
            94.501 * np.exp(-0.01292712 * soc)
            - 91.349 * np.exp(-0.01362893 * soc)
            + 1.472e-4 * soc ** 2
        )

        outputs["open_circuit_voltage"] = ocv

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        soc = inputs["state_of_charge"]

        partials["open_circuit_voltage", "state_of_charge"] = np.diag(
            -94.50 * 0.01292712 * np.exp(-0.01292712 * soc)
            + 91.349 * 0.01362893 * np.exp(-0.01362893 * soc)
            + 2.0 * 1.472e-4 * soc
        )

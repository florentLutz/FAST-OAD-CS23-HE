# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCurrents(om.ExplicitComponent):
    """
    Computation of the current going through the different components of the DC/DC converter,
    depends a lot on the duty cycle. See :cite:`hairik:2019` for the selected topology and the
    formula for the currents.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("duty_cycle", val=np.full(number_of_points, np.nan))
        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_output(
            "current_IGBT",
            val=np.full(number_of_points, 100.0),
            units="A",
            desc="Current going through the switch",
        )
        self.add_output(
            "current_capacitor",
            val=np.full(number_of_points, 100.0),
            units="A",
            desc="Current going through the filter capacitor",
        )
        self.add_output(
            "current_diode",
            val=np.full(number_of_points, 100.0),
            units="A",
            desc="Current going through the diode",
        )
        self.add_output(
            "current_inductor",
            val=np.full(number_of_points, 100.0),
            units="A",
            desc="Current going through the inductor",
        )

        self.declare_partials(of="current_IGBT", wrt="*", method="exact")
        self.declare_partials(of="current_capacitor", wrt="*", method="exact")
        self.declare_partials(of="current_diode", wrt="*", method="exact")
        self.declare_partials(of="current_inductor", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # We will clip the duty cycle to avoid any problems in the first loop of converging
        duty_cycle = np.clip(
            inputs["duty_cycle"],
            np.full_like(inputs["duty_cycle"], 1e-3),
            np.full_like(inputs["duty_cycle"], 1.0 - 1e-3),
        )
        current_out = inputs["dc_current_out"]

        outputs["current_IGBT"] = np.sqrt(duty_cycle) / (1.0 - duty_cycle) * current_out
        outputs["current_capacitor"] = np.sqrt(duty_cycle / (1.0 - duty_cycle)) * current_out
        outputs["current_diode"] = np.sqrt(1.0 / (1.0 - duty_cycle)) * current_out
        outputs["current_inductor"] = 1.0 / (1.0 - duty_cycle) * current_out

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        duty_cycle = inputs["duty_cycle"]
        current_out = inputs["dc_current_out"]

        partials["current_IGBT", "duty_cycle"] = (
            np.diag((duty_cycle + 1.0) / (2.0 * np.sqrt(duty_cycle) * (1.0 - duty_cycle) ** 2.0))
            * current_out
        )
        partials["current_IGBT", "dc_current_out"] = np.diag(
            np.sqrt(duty_cycle) / (1.0 - duty_cycle)
        )

        partials["current_capacitor", "duty_cycle"] = np.diag(
            current_out
            / (2.0 * np.sqrt(duty_cycle) * np.sqrt(1.0 - duty_cycle) * (1.0 - duty_cycle))
        )
        partials["current_capacitor", "dc_current_out"] = np.diag(
            np.sqrt(duty_cycle / (1.0 - duty_cycle))
        )

        partials["current_diode", "duty_cycle"] = np.diag(
            1.0
            / (2.0 * np.sqrt(1.0 / (1.0 - duty_cycle)))
            * 1.0
            / (1.0 - duty_cycle) ** 2.0
            * current_out
        )
        partials["current_diode", "dc_current_out"] = np.diag(np.sqrt(1.0 / (1.0 - duty_cycle)))

        partials["current_inductor", "duty_cycle"] = np.diag(
            1 / (1.0 - duty_cycle) ** 2.0 * current_out
        )
        partials["current_inductor", "dc_current_out"] = np.diag(1.0 / (1.0 - duty_cycle))

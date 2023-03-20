# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class PerformancesRectifierLoadSide(om.ExplicitComponent):
    """
    The rectifier is divided between a load side where the power source is and a generator side
    where the rest of the circuit is. This component represents the load side. We'll assume that
    at the input side of the rectifier, there are three AC phases

    Based on the methodology from :cite:`hendricks:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_rms_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="RMS value of the voltage at the input side of the rectifier",
        )
        self.add_input(
            "power",
            val=np.full(number_of_points, np.nan),
            units="W",
            desc="Power at the input side of the rectifier",
        )

        self.add_output(
            "ac_current_rms_in_one_phase",
            val=np.full(number_of_points, 200.0),
            units="A",
            desc="Current at the input side of the rectifier",
            lower=1e-4,
        )

        self.declare_partials(of="ac_current_rms_in_one_phase", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["ac_current_rms_in_one_phase"] = inputs["power"] / inputs["ac_voltage_rms_in"] / 3.0
        _LOGGER.debug("Load side voltage rms %s", str(inputs["ac_voltage_rms_in"]))

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["ac_current_rms_in_one_phase", "ac_voltage_rms_in"] = -np.diag(
            inputs["power"] / inputs["ac_voltage_rms_in"] ** 2 / 3.0
        )
        partials["ac_current_rms_in_one_phase", "power"] = np.diag(
            1.0 / inputs["ac_voltage_rms_in"] / 3.0
        )

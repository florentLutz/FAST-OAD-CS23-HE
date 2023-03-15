# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import logging

import openmdao.api as om
import numpy as np

_LOGGER = logging.getLogger(__name__)


class PerformancesSFC(om.ExplicitComponent):
    """
    Computation of the ICE sfc for the required torque, formula is based on :cite:`lutz:2022`
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("mean_effective_pressure", units="bar", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output(
            "specific_fuel_consumption", units="g/kW/h", val=200.0, shape=number_of_points
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Formula was established with dimensionless parameters, the division in the following
        # lines corresponds to the value used to render the value dimensionless. Also,
        # this correlation is valid for rpm between 2200.0 and 2700.0 and PME between 20.62 and
        # 8.08191746 bar, we will implement a simple and warning
        clipped_pme = np.clip(inputs["mean_effective_pressure"], 8.081, 20.62807778)
        if (clipped_pme != inputs["mean_effective_pressure"]).any():
            _LOGGER.warning(
                "Value of Mean Effective Pressure outside of the interpolation range, value clipped"
            )
        pme = clipped_pme / 20.62807778

        clipped_rpm = np.clip(inputs["rpm"], 2200.0, 2700.0)
        if (clipped_rpm != inputs["rpm"]).any():
            _LOGGER.warning("Value of RPM outside of the interpolation range, value clipped")
        rpm = clipped_rpm / 2700.0

        sfc = (
            -1663.04917125
            + 6507.41622273 * rpm
            + 829.44035557 * pme
            - 6292.54999479 * rpm ** 2.0
            - 4236.01419045 * pme * rpm
            + 1116.74781595 * pme ** 2.0
            + 2037.26111092 * rpm ** 3.0
            + 2217.65637191 * rpm ** 2.0 * pme
            + 253.52716382 * rpm * pme ** 2.0
            - 490.36976312 * pme ** 3.0
        )

        outputs["specific_fuel_consumption"] = sfc

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rpm = inputs["rpm"] / 2700.0
        pme = inputs["mean_effective_pressure"] / 20.62807778

        d_rpm_d_input_rpm = 1.0 / 2700.0
        d_pme_d_input_pme = 1.0 / 20.62807778

        partials["specific_fuel_consumption", "mean_effective_pressure"] = np.diag(
            (
                +829.44035557
                - 4236.01419045 * rpm
                + 2.0 * 1116.74781595 * pme
                + 2217.65637191 * rpm ** 2.0
                + 2.0 * 253.52716382 * rpm * pme
                - 3.0 * 490.36976312 * pme ** 2.0
            )
            * d_pme_d_input_pme
        )
        partials["specific_fuel_consumption", "rpm"] = np.diag(
            (
                +6507.41622273
                - 2.0 * 6292.54999479 * rpm
                - 4236.01419045 * pme
                + 3.0 * 2037.26111092 * rpm ** 2.0
                + 2.0 * 2217.65637191 * rpm * pme
                + 253.52716382 * pme ** 2.0
            )
            * d_rpm_d_input_rpm
        )

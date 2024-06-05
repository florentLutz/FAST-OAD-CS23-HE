# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesSFC(om.ExplicitComponent):
    """
    Computation of the ICE sfc for the required torque, formula is based on :cite:`lutz:2022`
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        ice_id = self.options["ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("mean_effective_pressure", units="bar", val=np.nan, shape=number_of_points)
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input(
            "settings:propulsion:he_power_train:ICE:" + ice_id + ":k_sfc",
            val=1.0,
            desc="K-factor to adjust the sfc of the ICE",
        )

        self.add_output(
            "specific_fuel_consumption", units="g/kW/h", val=200.0, shape=number_of_points
        )

        self.declare_partials(
            of="specific_fuel_consumption",
            wrt=["mean_effective_pressure", "rpm"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="specific_fuel_consumption",
            wrt="settings:propulsion:he_power_train:ICE:" + ice_id + ":k_sfc",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        # Formula was established with dimensionless parameters, the division in the following
        # lines corresponds to the value used to render the value dimensionless. Also,
        # this correlation is valid for rpm between 2200.0 and 2700.0 and PME between 20.62 and
        # 8.08191746 bar, we will implement a simple and warning
        clipped_pme = np.clip(inputs["mean_effective_pressure"], 8.081, 20.62807778)
        pme = clipped_pme / 20.62807778

        clipped_rpm = np.clip(inputs["rpm"], 2200.0, 2700.0)
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

        # The upper value is completely arbitrary, but is here to prevent the code looking for
        # absurdly high values
        sfc = np.clip(sfc, 0.0, 1.0e3)

        outputs["specific_fuel_consumption"] = (
            sfc * inputs["settings:propulsion:he_power_train:ICE:" + ice_id + ":k_sfc"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]

        rpm = inputs["rpm"] / 2700.0
        pme = inputs["mean_effective_pressure"] / 20.62807778

        d_rpm_d_input_rpm = 1.0 / 2700.0
        d_pme_d_input_pme = 1.0 / 20.62807778

        partials_mep = np.where(
            np.clip(inputs["mean_effective_pressure"], 8.081, 20.62807778)
            == inputs["mean_effective_pressure"],
            (
                +829.44035557
                - 4236.01419045 * rpm
                + 2.0 * 1116.74781595 * pme
                + 2217.65637191 * rpm ** 2.0
                + 2.0 * 253.52716382 * rpm * pme
                - 3.0 * 490.36976312 * pme ** 2.0
            )
            * d_pme_d_input_pme
            * inputs["settings:propulsion:he_power_train:ICE:" + ice_id + ":k_sfc"],
            1e-6,
        )

        partials["specific_fuel_consumption", "mean_effective_pressure"] = partials_mep

        partials["specific_fuel_consumption", "rpm"] = (
            (
                +6507.41622273
                - 2.0 * 6292.54999479 * rpm
                - 4236.01419045 * pme
                + 3.0 * 2037.26111092 * rpm ** 2.0
                + 2.0 * 2217.65637191 * rpm * pme
                + 253.52716382 * pme ** 2.0
            )
            * d_rpm_d_input_rpm
            * inputs["settings:propulsion:he_power_train:ICE:" + ice_id + ":k_sfc"]
        )
        partials[
            "specific_fuel_consumption",
            "settings:propulsion:he_power_train:ICE:" + ice_id + ":k_sfc",
        ] = (
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

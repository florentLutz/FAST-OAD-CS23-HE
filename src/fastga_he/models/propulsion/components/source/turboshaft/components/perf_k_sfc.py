#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesSpecificFuelConsumptionKFactorConstant(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_sfc",
            val=1.0,
            desc="K-factor to adjust the sfc/fuel consumption of the turboshaft",
        )

        self.add_output("k_sfc", val=1.0, shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(of="*", wrt="*", val=np.ones(number_of_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        outputs["k_sfc"] = np.full(
            number_of_points,
            inputs["settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_sfc"],
        )


class PerformancesSpecificFuelConsumptionKFactorVariable(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )
        self.options.declare(
            name="reference_rated_power",
            default=[300, 450],
            desc="Reference rated power for the correlation on the k_sfc",
        )
        self.options.declare(
            name="reference_k_sfc",
            default=[1.2, 1.0],
            desc="Reference sfc correction factor for the correlation on the k_sfc",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Flat rating of the turboshaft",
        )

        self.add_output("k_sfc", val=1.0, shape=number_of_points)
        self.add_output(
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_sfc",
            val=1.0,
            desc="K-factor to adjust the sfc/fuel consumption of the turboshaft",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]
        reference_rated_power = self.options["reference_rated_power"]
        reference_k_sfc = self.options["reference_k_sfc"]

        rated_power = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ].item()
        k_sfc = np.interp(rated_power, reference_rated_power, reference_k_sfc)

        outputs["k_sfc"] = np.full(number_of_points, k_sfc)
        outputs["settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_sfc"] = k_sfc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]
        reference_rated_power = self.options["reference_rated_power"]
        reference_k_sfc = self.options["reference_k_sfc"]

        rated_power = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ].item()

        derivative = (reference_k_sfc[-1] - reference_k_sfc[0]) / (
            reference_rated_power[-1] - reference_rated_power[0]
        )
        actual_derivative = (
            derivative
            if np.clip(rated_power, np.min(reference_rated_power), np.max(reference_rated_power))
            == rated_power
            else 0
        )

        partials[
            "k_sfc", "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ] = np.full(number_of_points, actual_derivative)
        partials[
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_sfc",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = actual_derivative

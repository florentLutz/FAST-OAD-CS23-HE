# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEquivalentRatedPowerITTLimit(om.ExplicitComponent):
    """
    Computation of the rated power equivalent to te thermodynamic power required if the ITT was a
    limit. Uses the definition of the thermodynamic power in the design point as a ratio of the
    rated power. The reason being the rated power is the constraint for the turboshaft.
    """

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
            "design_power_itt_limit",
            units="kW",
            val=np.nan,
            shape=number_of_points,
            desc="Thermodynamic power of the turboshaft at the design point if the ITT was "
            "limiting",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
            val=np.nan,
            desc="Ratio of the thermodynamic power divided by the rated power, typical values on "
            "the PT6A family is between 1.3 and 2.5",
        )

        self.add_output(
            "equivalent_rated_power_itt_limit",
            units="kW",
            val=750.0,
            shape=number_of_points,
            desc="Equivalent rated power of the turboshaft at the design point if the ITT was "
            "limiting",
        )

        self.declare_partials(
            of="equivalent_rated_power_itt_limit",
            wrt="design_power_itt_limit",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="equivalent_rated_power_itt_limit",
            wrt="data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        outputs["equivalent_rated_power_itt_limit"] = (
            inputs["design_power_itt_limit"]
            / inputs[
                "data:propulsion:he_power_train:turboshaft:"
                + turboshaft_id
                + ":design_point:power_ratio"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turboshaft_id = self.options["turboshaft_id"]
        number_of_points = self.options["number_of_points"]

        partials["equivalent_rated_power_itt_limit", "design_power_itt_limit"] = (
            np.ones(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:turboshaft:"
                + turboshaft_id
                + ":design_point:power_ratio"
            ]
        )
        partials[
            "equivalent_rated_power_itt_limit",
            "data:propulsion:he_power_train:turboshaft:"
            + turboshaft_id
            + ":design_point:power_ratio",
        ] = (
            -inputs["design_power_itt_limit"]
            / inputs[
                "data:propulsion:he_power_train:turboshaft:"
                + turboshaft_id
                + ":design_point:power_ratio"
            ]
            ** 2.0
        )

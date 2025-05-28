# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO
import logging
import numpy as np
import openmdao.api as om

_LOGGER = logging.getLogger(__name__)


class LCCPropellerCost(om.ExplicitComponent):
    """
    Computation of the propeller purchasing cost from :cite:`gudmundsson:2013`. The propeller
    type selection variable is only applied in cost estimation, does not affect other
    propeller-related calculations.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            units="lbf*ft",
            val=np.nan,
            desc="Maximum value of the propeller torque",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            units="min**-1",
            val=np.nan,
            desc="Maximum value of the propeller rpm",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":type",
            val=2.0,
            desc="Value set to 1.0 if fixed-pitch propeller, "
            "2.0 for constant-speed propeller. "
            "This is only use in cost estimation, does not affect other propeller-related "
            "models.",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
            units="USD",
            val=1000.0,
            desc="Unit purchase cost of the propeller",
        )

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":type",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        cpi_2012 = inputs["data:cost:cpi_2012"]
        rpm_rating = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating"
        ]
        torque_rating = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"
        ]
        d_prop = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        prop_type = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":type"]

        if prop_type == 1.0:
            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost"
            ] = 3145.0 * cpi_2012

        elif prop_type == 2.0:
            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost"
            ] = (
                209.69
                * cpi_2012
                * d_prop**2.0
                * (rpm_rating * torque_rating / d_prop / 5252.0) ** 0.12
            )

        else:
            outputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost"
            ] = 3145.0 * cpi_2012
            _LOGGER.warning("Propeller type %f does not exist, replaced by type 1.0!", prop_type)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        cpi_2012 = inputs["data:cost:cpi_2012"]
        rpm_rating = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating"
        ]
        torque_rating = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating"
        ]
        d_prop = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        prop_type = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":type"]

        if prop_type == 1.0:
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
                "data:cost:cpi_2012",
            ] = 3145.0

        elif prop_type == 2.0:
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
                "data:cost:cpi_2012",
            ] = 209.69 * d_prop**2.0 * (rpm_rating * torque_rating / d_prop / 5252.0) ** 0.12

            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":rpm_rating",
            ] = (
                25.1628
                * cpi_2012
                * d_prop**1.88
                * (torque_rating / 5252.0) ** 0.12
                / rpm_rating**0.88
            )

            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":torque_rating",
            ] = (
                25.1628
                * cpi_2012
                * d_prop**1.88
                * (rpm_rating / 5252.0) ** 0.12
                / torque_rating**0.88
            )

            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            ] = 394.2172 * cpi_2012 * d_prop**0.88 * (rpm_rating * torque_rating / 5252.0) ** 0.12

        else:
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":purchase_cost",
                "data:cost:cpi_2012",
            ] = 3145.0

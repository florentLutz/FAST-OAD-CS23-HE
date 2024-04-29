# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerThrustLoading(om.ExplicitComponent):
    """
    Adaptation of the formula taken from :cite:`de:2019`. Original name of this coefficient
    is the thrust coefficient. However, in order to not confuse it with the thrust coefficient
    from propeller performances, it will be named thrust loading.
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input("thrust", units="N", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=np.nan, shape=number_of_points)
        self.add_input("density", units="kg/m**3", val=np.nan, shape=number_of_points)

        self.add_output("thrust_loading", val=0.01, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        prop_dia = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        thrust = np.maximum(inputs["thrust"], np.ones_like(inputs["thrust"]))
        tas = inputs["true_airspeed"]
        rho = inputs["density"]

        thrust_loading = thrust / (rho * tas ** 2.0 * prop_dia ** 2.0)

        outputs["thrust_loading"] = thrust_loading

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        prop_dia = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        thrust = np.maximum(inputs["thrust"], np.ones_like(inputs["thrust"]))
        tas = inputs["true_airspeed"]
        rho = inputs["density"]

        partials[
            "thrust_loading",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            -2.0 * thrust / (rho * tas ** 2.0 * prop_dia ** 3.0)
        )
        partials["thrust_loading", "thrust"] = np.diag(1.0 / (rho * tas ** 2.0 * prop_dia ** 2.0))
        partials["thrust_loading", "true_airspeed"] = np.diag(
            -2.0 * thrust / (rho * tas ** 3.0 * prop_dia ** 2.0)
        )
        partials["thrust_loading", "density"] = np.diag(
            -thrust / (rho ** 2.0 * tas ** 2.0 * prop_dia ** 2.0)
        )

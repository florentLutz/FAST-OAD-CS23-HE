# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum power requested by the aux load.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):
        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "power_in",
            val=np.full(number_of_points, np.nan),
            units="W",
            desc="Power at the input side of the load",
        )
        self.add_input(
            "density",
            val=np.full(number_of_points, np.nan),
            units="kg/m**3",
        )
        self.add_input(
            name="waste_heat",
            val=np.full(number_of_points, np.nan),
            units="kW",
            desc="Heat from PEMFC to dissipate",
        )

        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            units="W",
            val=10e3,
        )
        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density",
            units="kg/m**3",
            val=1.2,
        )
        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max",
            units="kW",
            val=10.0,
        )

    def setup_partials(self):
        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            wrt="power_in",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density",
            wrt="density",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max",
            wrt="waste_heat",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max"] = np.max(
            inputs["power_in"]
        )
        outputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density"
        ] = np.min(inputs["density"])
        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max"] = (
            np.max(inputs["waste_heat"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            "power_in",
        ] = np.where(inputs["power_in"] == np.max(inputs["power_in"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density",
            "density",
        ] = np.where(inputs["density"] == np.min(inputs["density"]), 1.0, 0.0)

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":waste_heat_max",
            "waste_heat",
        ] = np.where(inputs["waste_heat"] == np.max(inputs["waste_heat"]), 1.0, 0.0)

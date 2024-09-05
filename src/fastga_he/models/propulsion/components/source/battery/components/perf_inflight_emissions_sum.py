# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBatteryPackInFlightEmissionsSum(om.ExplicitComponent):
    """
    Addition of the emissions of all pollutants at each step of the flight. Will be zero for
    batteries but still added for consistency with turboshaft and ICE.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input("CO2_emissions", val=np.full(number_of_points, np.nan), units="g")
        # For the LCA module we will adopt the following nomenclature:
        # "LCA" + phase + component + pollutant
        self.add_output(
            "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":CO2",
            units="g",
            val=3.1e5,
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":CO2",
            wrt="CO2_emissions",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

        self.add_input("CO_emissions", val=np.full(number_of_points, np.nan), units="g")
        self.add_output(
            "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":CO",
            units="g",
            val=0.8e4,
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":CO",
            wrt="CO_emissions",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

        self.add_input("NOx_emissions", val=np.full(number_of_points, np.nan), units="g")
        self.add_output(
            "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":NOx",
            units="g",
            val=3.1e2,
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":NOx",
            wrt="NOx_emissions",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

        self.add_input("SOx_emissions", val=np.full(number_of_points, np.nan), units="g")
        self.add_output(
            "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":SOx",
            units="g",
            val=0.4e2,
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":SOx",
            wrt="SOx_emissions",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

        self.add_input("H2O_emissions", val=np.full(number_of_points, np.nan), units="g")
        self.add_output(
            "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":H2O",
            units="g",
            val=1.2e5,
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":H2O",
            wrt="H2O_emissions",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

        self.add_input("HC_emissions", val=np.full(number_of_points, np.nan), units="g")
        self.add_output(
            "data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":HC",
            units="g",
            val=1.8e3,
        )
        self.declare_partials(
            of="data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":HC",
            wrt="HC_emissions",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs["data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":CO2"] = (
            np.sum(inputs["CO2_emissions"])
        )
        outputs["data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":CO"] = (
            np.sum(inputs["CO_emissions"])
        )
        outputs["data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":NOx"] = (
            np.sum(inputs["NOx_emissions"])
        )
        outputs["data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":SOx"] = (
            np.sum(inputs["SOx_emissions"])
        )
        outputs["data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":H2O"] = (
            np.sum(inputs["H2O_emissions"])
        )
        outputs["data:LCA:operation:he_power_train:battery_pack:" + battery_pack_id + ":HC"] = (
            np.sum(inputs["HC_emissions"])
        )

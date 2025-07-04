# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from ..constants import SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE


@oad.RegisterSubmodel(
    SUBMODEL_RECTIFIER_JUNCTION_TEMPERATURE,
    "fastga_he.submodel.propulsion.rectifier.junction_temperature.from_losses",
)
class PerformancesJunctionTemperature(om.ExplicitComponent):
    """
    Computation of the junction temperature in the diode and igbt module based on the losses on
    the previous point for each of those component. Assumes that the thermal constant is so small
    that we can consider that the losses equals the power we can dissipate in steady state.

    According to Semikron technical information, the heat transfer from junction to heat sink
    can be either modeled with a casing plate considered common to all junctions (consequently,
    a common R_th_cs for all modules), or consider individual R_th_cs. We will choose the former.

    Can be seen in :cite:`erroui:2019` and :cite:`tan:2022`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):
        rectifier_id = self.options["rectifier_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "casing_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the heat sink",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":igbt:thermal_resistance",
            units="K/W",
            val=np.nan,
            desc="Thermal resistance between the casing and the IGBT",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":diode:thermal_resistance",
            units="K/W",
            val=np.nan,
            desc="Thermal resistance between the casing and the heat sink",
        )

        self.add_input(
            "conduction_losses_diode",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input(
            "switching_losses_diode",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_input(
            "conduction_losses_IGBT",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input(
            "switching_losses_IGBT",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output(
            "diode_temperature",
            val=np.full(number_of_points, 273.15),
            units="degK",
            desc="Temperature of the diodes inside the module",
            shape=number_of_points,
        )
        self.add_output(
            "IGBT_temperature",
            val=np.full(number_of_points, 273.15),
            units="degK",
            desc="Temperature of the IGBTs inside the module",
            shape=number_of_points,
        )

        self.declare_partials(
            of="diode_temperature",
            wrt="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":diode:thermal_resistance",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="diode_temperature",
            wrt=["switching_losses_diode", "conduction_losses_diode"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="IGBT_temperature",
            wrt="data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":igbt:thermal_resistance",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="IGBT_temperature",
            wrt=["switching_losses_IGBT", "conduction_losses_IGBT"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="casing_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rectifier_id = self.options["rectifier_id"]

        temp_c = inputs["casing_temperature"]
        r_th_jc_igbt = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:thermal_resistance"
        ]
        r_th_jc_diode = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:thermal_resistance"
        ]

        # We assume that the IGBTs and diode dissipate their heat the same way inside the ,
        # so we will only look at the evolution in one IGBT and one diode
        diode_losses = inputs["switching_losses_diode"] + inputs["conduction_losses_diode"]
        igbt_losses = inputs["switching_losses_IGBT"] + inputs["conduction_losses_IGBT"]

        outputs["diode_temperature"] = temp_c + diode_losses * r_th_jc_diode
        outputs["IGBT_temperature"] = temp_c + igbt_losses * r_th_jc_igbt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]

        rectifier_id = self.options["rectifier_id"]

        r_th_jc_igbt = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:thermal_resistance"
        ]
        r_th_jc_diode = inputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":diode:thermal_resistance"
        ]

        # We assume that the IGBTs and diode dissipate their heat the same way inside the ,
        # so we will only look at the evolution in one IGBT and one diode
        diode_losses = inputs["switching_losses_diode"] + inputs["conduction_losses_diode"]
        igbt_losses = inputs["switching_losses_IGBT"] + inputs["conduction_losses_IGBT"]

        partials["diode_temperature", "switching_losses_diode"] = (
            np.ones(number_of_points) * r_th_jc_diode
        )
        partials["diode_temperature", "conduction_losses_diode"] = (
            np.ones(number_of_points) * r_th_jc_diode
        )
        partials[
            "diode_temperature",
            "data:propulsion:he_power_train:rectifier:"
            + rectifier_id
            + ":diode:thermal_resistance",
        ] = diode_losses

        partials["IGBT_temperature", "switching_losses_IGBT"] = (
            np.ones(number_of_points) * r_th_jc_igbt
        )
        partials["IGBT_temperature", "conduction_losses_IGBT"] = (
            np.ones(number_of_points) * r_th_jc_igbt
        )
        partials[
            "IGBT_temperature",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":igbt:thermal_resistance",
        ] = igbt_losses

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingInverterCasingThermalResistance(om.ExplicitComponent):
    """
    Computation of thermal resistances between the casing and the heat sink.

    According to Semikron technical information, the heat transfer from junction to heat sink can
    be either modeled with a casing plate considered common to all junctions (consequently,
    a common R_th_cs for all modules), or consider individual R_th_cs. We will choose the former.
    The consequence is that the thermal resistance will only depend on the size of the casing and
    since it is a constant shared between the IGBT7 modules considered, this modules will return
    a constant for now, but it might change hence the choice of keeping it as a component.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_input(
            name="data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
            units="A",
            val=np.nan,
            desc="Current caliber of one arm of the inverter",
        )

        self.add_output(
            name="data:propulsion:he_power_train:inverter:"
            + inverter_id
            + ":casing:thermal_resistance",
            units="K/W",
            val=0.014,
            desc="Thermal resistance between the casing and the heat sink",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance"
        ] = 0.010

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":casing:thermal_resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":current_caliber",
        ] = 0.0

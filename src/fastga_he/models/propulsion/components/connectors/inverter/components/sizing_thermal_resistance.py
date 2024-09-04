# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingInverterThermalResistances(om.ExplicitComponent):
    """
    Computation of thermal resistances of the diodes and IGBT, reference IGBT module for this is the
    SEMiX453GB12M7p.
    """

    def initialize(self):
        self.options.declare(
            name="inverter_id",
            default=None,
            desc="Identifier of the inverter",
            allow_none=False,
        )

        self.options.declare(
            name="R_th_igbt_ref",
            types=float,
            default=0.083,
            desc="Reference IGBT resistance (K/W)",
        )
        self.options.declare(
            name="R_th_diode_ref",
            types=float,
            default=0.107,
            desc="Reference diode resistance (K/W)",
        )

    def setup(self):
        inverter_id = self.options["inverter_id"]

        self.add_input(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:thermal_resistance",
            val=1e-3,
            units="K/W",
        )
        self.add_output(
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:thermal_resistance",
            val=1e-3,
            units="K/W",
        )

        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":igbt:thermal_resistance",
                "data:propulsion:he_power_train:inverter:"
                + inverter_id
                + ":diode:thermal_resistance",
            ],
            wrt="data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        inverter_id = self.options["inverter_id"]

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:thermal_resistance"
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance"]
            * self.options["R_th_igbt_ref"]
        )

        outputs[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:thermal_resistance"
        ] = (
            inputs["data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance"]
            * self.options["R_th_diode_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        inverter_id = self.options["inverter_id"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":igbt:thermal_resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
        ] = self.options["R_th_igbt_ref"]

        partials[
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":diode:thermal_resistance",
            "data:propulsion:he_power_train:inverter:" + inverter_id + ":scaling:resistance",
        ] = self.options["R_th_diode_ref"]

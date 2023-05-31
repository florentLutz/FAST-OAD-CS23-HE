# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingCapacitorThermalResistance(om.ExplicitComponent):
    """
    Computation of the capacitor's thermal resistance. Implementation of the workflow from
    :cite:`budinger_sizing_2023`.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use a capacitor",
            allow_none=False,
        )
        self.options.declare(
            name="r_th_ref",
            types=float,
            default=3.0,
            desc="Thermal resistance of the reference component [degK/W]",
        )

    def setup(self):

        prefix = self.options["prefix"]

        self.add_input(
            name=prefix + ":capacitor:scaling:thermal_resistance",
            val=np.nan,
            desc="Scaling factor for the capacitor thermal resistance",
        )

        self.add_output(
            name=prefix + ":capacitor:thermal_resistance",
            units="degK/W",
            val=self.options["r_th_ref"],
            desc="Capacitor's thermal resistance",
        )

        self.declare_partials(of="*", wrt="*", val=self.options["r_th_ref"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prefix = self.options["prefix"]

        outputs[prefix + ":capacitor:thermal_resistance"] = (
            self.options["r_th_ref"] * inputs[prefix + ":capacitor:scaling:thermal_resistance"]
        )

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_RECTIFIER_EFFICIENCY

oad.RegisterSubmodel.active_models[
    SUBMODEL_RECTIFIER_EFFICIENCY
] = "fastga_he.submodel.propulsion.rectifier.efficiency.from_losses"


@oad.RegisterSubmodel(
    SUBMODEL_RECTIFIER_EFFICIENCY, "fastga_he.submodel.propulsion.rectifier.efficiency.from_losses"
)
class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency of the rectifier."""

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

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the output side of the rectifier",
        )
        self.add_input(
            "dc_current_out",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the output side of the rectifier",
        )
        self.add_input(
            "losses_rectifier",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )

        self.add_output("efficiency", val=0.98, shape=number_of_points, lower=0.0, upper=1.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_current_out = inputs["dc_current_out"]
        dc_voltage_out = inputs["dc_voltage_out"]
        losses_rectifier = inputs["losses_rectifier"]

        useful_power = dc_current_out * dc_voltage_out

        eta = np.where(useful_power < 10.0, 1.0, useful_power / (useful_power + losses_rectifier))

        outputs["efficiency"] = eta

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_current_out = inputs["dc_current_out"]
        dc_voltage_out = inputs["dc_voltage_out"]
        losses_rectifier = inputs["losses_rectifier"]

        useful_power = dc_current_out * dc_voltage_out

        partials_dc_current = np.where(
            useful_power < 10.0,
            1e-6,
            losses_rectifier / (useful_power + losses_rectifier) ** 2.0 * dc_voltage_out,
        )
        partials["efficiency", "dc_current_out"] = np.diag(partials_dc_current)

        partials_dc_voltage = np.where(
            useful_power < 10.0,
            1e-6,
            losses_rectifier / (useful_power + losses_rectifier) ** 2.0 * dc_current_out,
        )
        partials["efficiency", "dc_voltage_out"] = np.diag(partials_dc_voltage)

        partials_losses = np.where(
            useful_power < 10.0,
            1e-6,
            -useful_power / (useful_power + losses_rectifier) ** 2.0,
        )
        partials["efficiency", "losses_rectifier"] = np.diag(partials_losses)

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2024  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import openmdao.api as om

from .turboshaft_off_design_fuel import Turboshaft


####################################################################################################


class TurboshaftMaxPowerOPRLimit(Turboshaft):
    def setup(self):

        n = self.options["number_of_points"]

        self.add_subsystem(
            "distance_to_limit_opr_limit",
            DistanceToLimitOPRLimit(number_of_points=n),
            promotes=["*"],
        )

        super().setup()


class DistanceToLimitOPRLimit(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("opr", shape=n, val=np.nan)
        self.add_input("opr_limit", shape=n, val=np.nan)

        self.add_output("required_shaft_power", units="kW", val=np.full(n, 500.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        opr = inputs["opr"]
        opr_limit = inputs["opr_limit"]

        residuals["required_shaft_power"] = opr / opr_limit - 1.0

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        opr = inputs["opr"]
        opr_limit = inputs["opr_limit"]

        jacobian["required_shaft_power", "opr"] = np.diag(1.0 / opr_limit)
        jacobian["required_shaft_power", "opr_limit"] = np.diag(-opr / opr_limit ** 2.0)

        jacobian["required_shaft_power", "required_shaft_power"] = np.diag(np.zeros_like(opr))


####################################################################################################


class TurboshaftMaxPowerITTLimit(Turboshaft):
    def setup(self):

        n = self.options["number_of_points"]

        self.add_subsystem(
            "distance_to_limit_itt_limit",
            DistanceToLimitITTLimit(number_of_points=n),
            promotes=["*"],
        )

        super().setup()


class DistanceToLimitITTLimit(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_temperature_45", units="degK", shape=n, val=np.nan)
        self.add_input("itt_limit", units="degK", shape=n, val=np.nan)

        self.add_output("required_shaft_power", units="kW", val=np.full(n, 500.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        total_temperature_45 = inputs["total_temperature_45"]
        itt_limit = inputs["itt_limit"]

        residuals["required_shaft_power"] = total_temperature_45 / itt_limit - 1.0

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        total_temperature_45 = inputs["total_temperature_45"]
        itt_limit = inputs["itt_limit"]

        jacobian["required_shaft_power", "total_temperature_45"] = np.diag(1.0 / itt_limit)
        jacobian["required_shaft_power", "itt_limit"] = np.diag(
            -total_temperature_45 / itt_limit ** 2.0
        )

        jacobian["required_shaft_power", "required_shaft_power"] = np.diag(
            np.zeros_like(total_temperature_45)
        )

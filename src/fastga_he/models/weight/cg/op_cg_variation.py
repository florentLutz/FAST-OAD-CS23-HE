#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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


class OperationalInFlightCGVariation(om.ExplicitComponent):
    """
    Computes the coefficient necessary to the calculation of the cg position at any point of
    the operational mission.
    """

    def setup(self):
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:mass", val=np.nan, units="kg")
        self.add_input("data:mission:operational:payload:mass", val=np.nan, units="kg")
        self.add_input("data:mission:operational:payload:CG:x", val=np.nan, units="m")

        self.add_output(
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment",
            units="kg*m",
        )
        self.add_output(
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:mass", units="kg"
        )

        self.declare_partials(
            of="data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:mass",
            wrt=["data:mission:operational:payload:mass", "data:weight:aircraft_empty:mass"],
            val=1.0,
        )
        self.declare_partials(
            of="data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment",
            wrt=[
                "data:weight:aircraft_empty:mass",
                "data:weight:aircraft_empty:CG:x",
                "data:mission:operational:payload:mass",
                "data:mission:operational:payload:CG:x",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_empty = inputs["data:weight:aircraft_empty:mass"]
        x_cg_plane_aft = inputs["data:weight:aircraft_empty:CG:x"]

        payload = inputs["data:mission:operational:payload:mass"]
        x_cg_payload = inputs["data:mission:operational:payload:CG:x"]

        equivalent_moment = m_empty * x_cg_plane_aft + payload * x_cg_payload
        mass = m_empty + payload

        outputs[
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment"
        ] = equivalent_moment
        outputs["data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:mass"] = mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment",
            "data:weight:aircraft_empty:mass",
        ] = inputs["data:weight:aircraft_empty:CG:x"]
        partials[
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment",
            "data:weight:aircraft_empty:CG:x",
        ] = inputs["data:weight:aircraft_empty:mass"]
        partials[
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment",
            "data:mission:operational:payload:mass",
        ] = inputs["data:mission:operational:payload:CG:x"]
        partials[
            "data:weight:aircraft:in_flight_variation:operational:fixed_mass_comp:equivalent_moment",
            "data:mission:operational:payload:CG:x",
        ] = inputs["data:mission:operational:payload:mass"]

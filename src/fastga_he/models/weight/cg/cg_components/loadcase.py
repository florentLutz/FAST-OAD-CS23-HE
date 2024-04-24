# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastga.models.weight.cg.cg_components.constants import SUBMODEL_LOADCASE_FLIGHT_X


@oad.RegisterSubmodel(
    SUBMODEL_LOADCASE_FLIGHT_X, "fastga_he.submodel.weight.cg.loadcase.flight.simple"
)
class ComputeFlightCGCase(om.ExplicitComponent):
    """Center of gravity estimation for all load cases in flight"""

    def initialize(self):

        # Not used but required for compatibility reasons
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:CG:fwd:MAC_limit",
            val=np.nan,
            units="m",
            desc="position of the fwd limit of the Weight and balance envelop as a percent of MAC",
        )
        self.add_input(
            "data:weight:aircraft:CG:aft:MAC_limit",
            val=np.nan,
            units="m",
            desc="position of the aft limit of the Weight and balance envelop as a percent of MAC",
        )

        self.add_output("data:weight:aircraft:CG:flight_condition:max:MAC_position", val=0.40)
        self.add_output("data:weight:aircraft:CG:flight_condition:min:MAC_position", val=0.15)

        self.declare_partials(
            of="data:weight:aircraft:CG:flight_condition:max:MAC_position",
            wrt=[
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:at25percent:x",
                "data:weight:aircraft_empty:CG:x",
                "data:weight:aircraft:CG:aft:MAC_limit",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:weight:aircraft:CG:flight_condition:min:MAC_position",
            wrt=[
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:at25percent:x",
                "data:weight:aircraft_empty:CG:x",
                "data:weight:aircraft:CG:fwd:MAC_limit",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_aircraft_empty = inputs["data:weight:aircraft_empty:CG:x"]

        aft_limit = inputs["data:weight:aircraft:CG:aft:MAC_limit"]
        fwd_limit = inputs["data:weight:aircraft:CG:fwd:MAC_limit"]

        cg_aircraft_empty_mac = (cg_aircraft_empty - fa_length + 0.25 * l0_wing) / l0_wing

        outputs["data:weight:aircraft:CG:flight_condition:min:MAC_position"] = (
            cg_aircraft_empty_mac - fwd_limit
        )
        outputs["data:weight:aircraft:CG:flight_condition:max:MAC_position"] = (
            cg_aircraft_empty_mac + aft_limit
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_aircraft_empty = inputs["data:weight:aircraft_empty:CG:x"]

        partials[
            "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            "data:geometry:wing:MAC:at25percent:x",
        ] = (
            -1.0 / l0_wing
        )
        partials[
            "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            "data:geometry:wing:MAC:length",
        ] = (
            -(cg_aircraft_empty - fa_length) / l0_wing ** 2.0
        )
        partials[
            "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            "data:weight:aircraft_empty:CG:x",
        ] = (
            1.0 / l0_wing
        )
        partials[
            "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            "data:weight:aircraft:CG:fwd:MAC_limit",
        ] = -1.0

        partials[
            "data:weight:aircraft:CG:flight_condition:max:MAC_position",
            "data:geometry:wing:MAC:at25percent:x",
        ] = (
            -1.0 / l0_wing
        )
        partials[
            "data:weight:aircraft:CG:flight_condition:max:MAC_position",
            "data:geometry:wing:MAC:length",
        ] = (
            -(cg_aircraft_empty - fa_length) / l0_wing ** 2.0
        )
        partials[
            "data:weight:aircraft:CG:flight_condition:max:MAC_position",
            "data:weight:aircraft_empty:CG:x",
        ] = (
            1.0 / l0_wing
        )
        partials[
            "data:weight:aircraft:CG:flight_condition:max:MAC_position",
            "data:weight:aircraft:CG:aft:MAC_limit",
        ] = 1.0

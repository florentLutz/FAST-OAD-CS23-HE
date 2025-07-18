# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


from ..components.slipstream_delta_cl_group import SlipstreamPropellerDeltaClGroup
from ..components.slipstream_delta_cd0 import SlipstreamPropellerDeltaCD0
from ..components.slipstream_delta_cm0 import SlipstreamPropellerDeltaCM0
from ..components.slipstream_delta_cm_alpha import SlipstreamPropellerDeltaCMAlpha
from ..components.slipstream_delta_cm import SlipstreamPropellerDeltaCM


class SlipstreamPropeller(om.Group):
    """
    Adaptation of the methodology from :cite:`de:2019` to compute the delta caused by the blowing
    of the propeller. Does not yet include the delta_Cdi as, as is discussed in other component,
    the total delta_Cl is required for it to be mathematically correct.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
            values=["cruise", "takeoff", "landing"],
        )
        self.options.declare(
            "low_speed_aero",
            default=False,
            desc="Boolean to consider low speed aerodynamics",
            types=bool,
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]
        low_speed_aero = self.options["low_speed_aero"]

        self.add_subsystem(
            name="delta_cl_at_AoA",
            subsys=SlipstreamPropellerDeltaClGroup(
                propeller_id=propeller_id,
                number_of_points=number_of_points,
                flaps_position=flaps_position,
                low_speed_aero=low_speed_aero,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cd0",
            subsys=SlipstreamPropellerDeltaCD0(
                propeller_id=propeller_id,
                number_of_points=number_of_points,
                flaps_position=flaps_position,
                low_speed_aero=low_speed_aero,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cm0",
            subsys=SlipstreamPropellerDeltaCM0(
                propeller_id=propeller_id,
                number_of_points=number_of_points,
                flaps_position=flaps_position,
                low_speed_aero=low_speed_aero,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cm_alpha",
            subsys=SlipstreamPropellerDeltaCMAlpha(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cm",
            subsys=SlipstreamPropellerDeltaCM(
                number_of_points=number_of_points,
            ),
            promotes=["*"],
        )

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingDCBus(om.Group):
    """
    Class that regroups all of the sub components for the sizing of the DC Bus.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the AC bus, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        ac_bus_id = self.options["ac_bus_id"]
        position = self.options["position"]

        ivc_sizing = om.IndepVarComp()
        ivc_sizing.add_output(
            name="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":mass",
            units="kg",
            val=0.0,
            desc="Weight of the bus bar",
        )
        ivc_sizing.add_output(
            name="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the DC bus center of gravity",
        )
        ivc_sizing.add_output(
            name="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":low_speed:CD0",
            val=0.0,
        )
        ivc_sizing.add_output(
            name="data:propulsion:he_power_train:AC_bus:" + ac_bus_id + ":cruise:CD0",
            val=0.0,
        )

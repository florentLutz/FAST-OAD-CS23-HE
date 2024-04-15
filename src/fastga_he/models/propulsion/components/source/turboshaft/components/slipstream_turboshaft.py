# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from .slipstream_mach import SlipstreamMach
from .slipstream_density_ratio import SlipstreamDensityRatio
from .slipstream_required_power import SlipstreamRequiredPower
from .slipstream_exhaust_mass_flow import SlipstreamExhaustMassFlow
from .slipstream_exhaust_velocity import SlipstreamExhaustVelocity
from .slipstream_exhaust_thrust import SlipstreamExhaustThrust
from .slipstream_delta_cd import SlipstreamTurboshaftDeltaCd


class SlipstreamTurboshaft(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        self.add_subsystem(
            name="density_ratio",
            subsys=SlipstreamDensityRatio(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="mach",
            subsys=SlipstreamMach(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="required_power",
            subsys=SlipstreamRequiredPower(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="exhaust_mass_flow",
            subsys=SlipstreamExhaustMassFlow(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="exhaust_velocity",
            subsys=SlipstreamExhaustVelocity(
                number_of_points=number_of_points, turboshaft_id=turboshaft_id
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="exhaust_thrust",
            subsys=SlipstreamExhaustThrust(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_Cd",
            subsys=SlipstreamTurboshaftDeltaCd(number_of_points=number_of_points),
            promotes=["*"],
        )

        ivc = om.IndepVarComp()

        ivc.add_output("delta_Cl", val=np.zeros(number_of_points))
        ivc.add_output("delta_Cm", val=np.zeros(number_of_points))

        self.add_subsystem(name="deltas", subsys=ivc, promotes=["*"])

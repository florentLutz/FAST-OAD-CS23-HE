# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SlipstreamH2FuelSystem(om.Group):
    """
    Computation of the variation of aerodynamic coefficient during the mission. This component is
    required as all components are required to provide one component that computes the delta's
    but all expect the propeller will be 0.0 at the beginning.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        ivc = om.IndepVarComp()

        ivc.add_output("delta_Cl", val=np.zeros(number_of_points))
        ivc.add_output("delta_Cd", val=np.zeros(number_of_points))
        ivc.add_output("delta_Cm", val=np.zeros(number_of_points))

        self.add_subsystem(name="deltas", subsys=ivc, promotes=["*"])

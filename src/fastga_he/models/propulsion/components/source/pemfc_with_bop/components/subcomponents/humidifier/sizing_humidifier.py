# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingHumidifier(om.Group):
    """
    Sizing of the PEMFC Humidifier
    """

    def setup(self):
        ## Inputs
        self.add_input(name="data:thermal:fuel_cell:power", units="kW", val=np.nan)

        self.add_output(name="data:thermal:humidifier:mass", units="kg")
        self.add_output(name="data:thermal:humidifier:volume", units="m**3")

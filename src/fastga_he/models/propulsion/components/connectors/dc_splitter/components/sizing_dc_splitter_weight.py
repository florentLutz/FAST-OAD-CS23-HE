# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om


class SizingDCSplitterWeight(om.ExplicitComponent):
    """
    Computation of the weight of a DC splitter. Not a real computation, since it will always be
    assumed to be "glued" to a bus, and it shall just acts as a component to distribute power
    among several sources :cite:`brelje:2018`.
    """

    def initialize(self):
        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )

    def setup(self):

        dc_splitter_id = self.options["dc_splitter_id"]

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":mass",
            units="kg",
            val=0.0,
            desc="Mass of the DC splitter",
        )

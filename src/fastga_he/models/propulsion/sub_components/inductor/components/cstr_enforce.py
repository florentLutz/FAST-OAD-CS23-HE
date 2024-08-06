# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from ..constants import SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP


import openmdao.api as om
import numpy as np

import fastoad.api as oad


@oad.RegisterSubmodel(
    SUBMODEL_CONSTRAINTS_INDUCTOR_AIR_GAP,
    "fastga_he.submodel.propulsion.constraints.inductor.air_gap.enforce",
)
class ConstraintsInductorAirGapEnforce(om.ExplicitComponent):
    """
    Class that enforces that the air gap chosen for the sizing of the inductor inside the
    DC/DC converter is equal to the maximum allowed.
    """

    def initialize(self):
        self.options.declare(
            name="prefix",
            default=None,
            desc="Prefix for the components that will use an inductor",
            allow_none=False,
        )

    def setup(self):
        prefix = self.options["prefix"]

        self.add_input(
            prefix + ":inductor:core_dimension:C",
            units="m",
            val=np.nan,
            desc="C dimension of the E-core in the inductor",
        )
        self.add_output(
            name=prefix + ":inductor:air_gap",
            units="m",
            val=2.75e-3,
            desc="Air gap in the inductor",
        )
        self.declare_partials(
            of=prefix + ":inductor:air_gap",
            wrt=prefix + ":inductor:core_dimension:C",
            val=0.1,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prefix = self.options["prefix"]

        outputs[prefix + ":inductor:air_gap"] = 0.1 * inputs[prefix + ":inductor:core_dimension:C"]

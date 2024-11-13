# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCACoreAggregation(om.ExplicitComponent):
    """
    We'll use the same feature of OpenMDAO as for the LCACoreNormalization.

    @felixpollet: for this component, we will assume the eutrophication of oranges is out of the
    scope.
    """

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:environmental_impact:single_score"] = np.sum(inputs.values())

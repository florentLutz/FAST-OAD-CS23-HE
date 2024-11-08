# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from .lca_core import LCA_PREFIX


class LCACoreNormalisation(om.ExplicitComponent):
    """
    We'll use here a particularity of the OpenMDAO framework. We actually can't have to define a
    setup function right away here because we don't (and can't) know the inputs, since they are
    outputs of the LCACore components and that thy will be defined whn running the code after
    parsing the LCA Configuration file (We could parse it here but it will take too long).

    Instead, as suggested by the magnificent @felixpollet, we will use the configure function of the
    parent group which runs after the setup of all subsystem of that parent group.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Will be changed by the configure method of the parent group eventually.
        self.clean_method_name = None

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pass

# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om

from .resources.constants import LCA_PREFIX


class LCACoreWeighting(om.ExplicitComponent):
    """
    We'll use the same feature of OpenMDAO as for the LCACoreNormalization.

    Hello @felixpollet, I know you are reading this. Stop stalking me and go back to work :p
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Will be changed by the configure method of the parent group eventually.
        self.inputs_list = None
        self.weighting_factor = None

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for var_in in self.inputs_list:
            normalized_method_name = var_in.split(":")[2]
            method_name = normalized_method_name.replace("_normalized", "")
            if method_name in self.weighting_factor:
                weighted_method_name = method_name + "_weighted"
                weighting_factor_name = LCA_PREFIX + method_name + ":weighting_factor"
                weighting_factor = inputs[weighting_factor_name]

                var_out = var_in.replace(normalized_method_name, weighted_method_name)

                outputs[var_out] = inputs[var_in] * weighting_factor

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for var_in in self.inputs_list:
            normalized_method_name = var_in.split(":")[2]
            method_name = normalized_method_name.replace("_normalized", "")
            if method_name in self.weighting_factor:
                weighted_method_name = method_name + "_weighted"
                weighting_factor_name = LCA_PREFIX + method_name + ":weighting_factor"

                var_out = var_in.replace(normalized_method_name, weighted_method_name)

                partials[var_out, var_in] = inputs[weighting_factor_name]
                partials[var_out, weighting_factor_name] = inputs[var_in]

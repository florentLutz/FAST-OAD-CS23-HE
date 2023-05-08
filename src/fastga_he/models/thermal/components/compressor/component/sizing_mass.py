import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeCompressorMass(om.ExplicitComponent):
    """
    Sizes the compressor mass
    """

    def setup(self):

        ## Inputs
        self.add_input(name="data:thermal:compressor:power", units="W", val=np.nan)

        ## Outputs
        self.add_output(name="data:thermal:compressor:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        P = inputs["data:thermal:compressor:power"]

        M = 0.0400683 * (P / 1000) + 5.17242

        outputs["compressor:mass"] = M

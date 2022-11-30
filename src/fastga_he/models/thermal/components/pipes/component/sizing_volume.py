import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputePipeVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:pipes:distance",
            units="m",
            val=np.nan,
            desc="total distance of pipes in TMS",
        )

        self.add_input(
            name="data:thermal:pipes:radius",
            units="m",
            val=np.nan,
            desc="pipe radiuss",
        )

        self.add_output(name="data:thermal:pipes:coolant:volume", units="m**3")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        d_pipe = inputs["data:thermal:pipes:distance"]
        R_pipe = inputs["data:thermal:pipes:radius"]

        V_coolant_pipe = np.pi * d_pipe * R_pipe**2

        outputs["data:thermal:pipes:coolant:volume"] = V_coolant_pipe

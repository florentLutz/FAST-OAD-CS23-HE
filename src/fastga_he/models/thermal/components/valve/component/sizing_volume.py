import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeValvzVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:coolant:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="coolant mass flow",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_flow = inputs["data:thermal:coolant:mass_flow"]

        V = 6e-05 * m_flow + 8e-05

        outputs["data:thermal:valvevolume"] = V

import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeValvzVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:coolant:mass:flow",
            units="m",
            val=np.nan,
            desc="coolant mass flow",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_flow = inputs["data:thermal:coolant:mass_flow"]

        if m_flow
        V_valve = 0.0003*m_flow-0.0001 

        outputs["data:thermal:valvevolume"] = V_valve

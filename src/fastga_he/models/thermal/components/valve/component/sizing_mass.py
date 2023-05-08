import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputValveMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:coolant:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="coolant mass flow",
        )

        self.add_output(name="data:thermal:valve:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_flow = inputs["data:thermal:coolant:mass_flow"]

        M = 0.568 * (m_flow**0.55410)

        outputs["data:thermal:valve:mass"] = M

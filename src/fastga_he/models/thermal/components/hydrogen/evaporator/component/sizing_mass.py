import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeHydrogenEvaporatorMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:hydrogen:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="required mass flow of hydrogen",
        )

        self.add_output(name="data:thermal:hydrogen:evaporator:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        M_H2_evaporator = 
        outputs["data:thermal:hydrogen:evaporator:mass"] = M_H2_evaporator

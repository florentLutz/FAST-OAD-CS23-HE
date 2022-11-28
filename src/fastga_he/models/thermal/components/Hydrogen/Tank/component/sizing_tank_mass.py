import numpy as np
import scipy.constants as sc
import openmdao.api as om

class ComputeHydrogenTank(om.ExplicitComponent):
    
    def setup(self):

        self.add_input(name="data:thermal:hydrogen:mass_flow", units = "kg/s", val=np.nan)
        self.add_input(name="data:thermal:hydrogen:tank:gravimetric_efficiency", val=0.5)
        self.add_input(name="data:thermal:hydrogen:tank:volumetric_efficiency", val=np.nan)
        self.add_input(name="time", units="s", val=np.nan)

        self.add_output(name="data:thermal:hydrogen:tank:mass", units="kg")
        self.add_output(name="data:thermal:hydrogen:tank:volume", units="m**3")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        M_H2 = inputs["data:thermal:hydrogen:mass_flow"]*inputs["time"]
        n_g = inputs["data:thermal:hydrogen:tank:gravimetric_efficiency"]
        n_v = inputs["data:thermal:hydrogen:tank:volumetric_efficiency"]

        M_H2_tank = M_H2/n_g - M_H2
        V_H2_tank = 0.0366*M_H2 + 0.1215

        outputs["data:thermal:hydrogen:tank:mass"] = M_H2_tank
        outputs["data:thermal:hydrogen:tank:volume"] = V_H2_tank
import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeHydrogenTankVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:hydrogen:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="required mass flow of hydrogen",
        )

        self.add_output(name="data:thermal:hydrogen:tank:volume", units="m**3")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        M_H2 = inputs["data:thermal:hydrogen:mass_flow"] * inputs["time"]

        V_H2_tank = 0.0366 * M_H2 + 0.1215

        outputs["data:thermal:hydrogen:tank:volume"] = V_H2_tank

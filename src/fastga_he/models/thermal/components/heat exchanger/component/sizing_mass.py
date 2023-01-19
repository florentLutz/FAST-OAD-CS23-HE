import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeHEXMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:HEX:volume",
            units="m**3",
            val=np.nan,
            desc="HEX volume",
        )
        self.add_input(
            name="data:thermal:HEX:unit_density",
            units="kg/m**3",
            val=np.nan,
            desc="HEX unit density",
        )

        self.add_output(
            name="data:thermal:HEX:mass",
            units="kg"
            desc="HEX mass"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        volume = inputs["data:thermal:HEX:volume"]
        rho = inputs["data:thermal:HEX:unit_density"]

        mass = volume*rho

        outputs["data:thermal:HEX:mass"] = mass

import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeEvaporatorMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:evaporator:volume",
            units="m**3",
            val=np.nan,
            desc="evaporator volume",
        )
        self.add_input(
            name="data:thermal:evaporator:unit_density",
            units="kg/m**3",
            val=np.nan,
            desc="evaporator unit density",
        )

        self.add_output(
            name="data:thermal:evaporator:mass",
            units="kg"
            desc="evaporator mass"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        volume = inputs["data:thermal:evaporator:volume"]
        rho = inputs["data:thermal:evaporator:unit_density"]

        mass = volume*rho

        outputs["data:thermal:evaporator:mass"] = mass

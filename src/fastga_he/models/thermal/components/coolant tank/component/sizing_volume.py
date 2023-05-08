import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeCoolantTankVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:coolant:mass",
            units="kg",
            val=np.nan,
            desc="coolant mass",
        )

        self.add_input(
            name="data:thermal:coolant:density",
            units="kg/m**3",
            val=np.nan,
            desc="density of coolant",
        )

        self.add_output(name="data:thermal:coolant_tank:volume", units="m**3")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        M_cool = inputs["data:thermal:coolant:mass"]
        rho_cool = inputs["data:thermal:coolant:density"]

        V = M_cool / rho_cool

        outputs["data:thermal:coolant_tank:volume"] = V

import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeCoolantTankMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:coolant_tank:volume",
            units="m**3",
            val=np.nan,
            desc="coolant tank volume",
        )

        self.add_input(
            name="settings:thermal:coolant_tank:material:density",
            units="kg/m**3"
            val=2640 ,
            desc="material density of the coolant tank"
        )

        self.add_output(name="data:thermal:pipes:coolant_tank:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        V_tank = inputs["data:thermal:coolant_tank:volume"]
        rho_tank = inputs["data:thermal:coolant_tank:material:density"]

        M_tank = rho_tank*V_tank

        outputs["data:thermal:coolant_tank:mass"] = M_tank


import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeCoolantMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:pipes:radius",
            units="m",
            val=np.nan,
            desc="pipe radius",
        )

        self.add_input(
            name="data:thermal:pipes:distance",
            units="m",
            val=np.nan,
            desc="total distance of pipes in TMS",
        )

        self.add_input(
            name="data:thermal:pipes:thickness",
            units="m",
            val=1e-3,
            desc="pipe thickness",
        )

        self.add_input(
            name="data:thermal:coolant:density",
            units="kg/m**3",
            val=np.nan,
            desc="density of coolant",
        )

        self.add_output(name="data:thermal:coolant:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rho_coolant = inputs["data:thermal:coolant:density"]
        R_pipe = inputs["data:thermal:pipes:radius"]
        d_pipe = inputs["data:thermal:pipes:distance"]
        t_pipe = inputs["data:thermal:pipes:thickness"]

        M_coolant = rho_coolant * np.pi * d_pipe * [(R_pipe + t_pipe) ** 2 - R_pipe**2]

        outputs["data:thermal:coolant:mass"] = M_coolant

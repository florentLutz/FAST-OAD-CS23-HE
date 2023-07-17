import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputePipePressureLoss(om.ExplicitComponent):
    def setup(self):
        self.add_input(
            name="data:thermal:coolant:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="coolant mass flow",
        )

        self.add_input(
            name="data:thermal:pipes:radius",
            units="m",
            val=np.nan,
            desc="coolant pipes radius",
        )

        self.add_input(
            name="data:thermal:coolant:density",
            units="kg/m**3",
            val=np.nan,
            desc="coolant density",
        )

        self.add_input(
            name="data:thermal:coolant:dynamic_viscosity",
            units="Pa*s",
            val=np.nan,
            desc="coolant dynamic viscosity",
        )

        self.add_input(
            name="data:thermal:pipes:distance",
            units="m",
            val=np.nan,
            desc="total distance of pipes in TMS",
        )

        self.add_output(name="data:thermal:pipes:pressure_loss", units="Pa")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_flow = inputs["data:thermal:coolant:mass_flow"]
        rho_coolant = inputs["data:thermal:coolant:density"]
        mu_coolant = inputs["data:thermal:coolant:dynamic_viscosity"]
        d_pipe = inputs["data:thermal:pipes:distance"]
        R_pipe = inputs["data:thermal:pipes:radius"]

        D_H = 4 * np.pi * R_pipe**2 / 2 / np.pi / R_pipe
        v_avg_cool = m_flow / rho_coolant / np.pi / R_pipe**2

        Re = rho_coolant * v_avg_cool * D_H / mu_coolant
        f_d = (2.7 * Re ** (-0.68)) ** (1 / (1 + (Re / 2000) ** 9)) * (0.13 * Re ** (-0.28)) ** (
            1 - 1 / (1 + (Re / 2000) ** 9)
        )

        dp_pipes = f_d * d_pipe / D_H * rho_coolant * v_avg_cool**2 / 2

        outputs["data:thermal:pipes:pressure_loss"] = dp_pipes

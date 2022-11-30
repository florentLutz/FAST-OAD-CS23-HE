import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputePipeMass(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:coolant:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="coolant mass flow",
        )

        self.add_input(
            name="data:thermal:coolant:velocity",
            units="m/s",
            val=np.nan,
            desc="maximum coolant velocity in liquid state",
        )

        self.add_input(
            name="data:thermal:coolant:density",
            units="kg/m**3",
            val=np.nan,
            desc="coolant density",
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
            name="data:thermal:pipes:density",
            units="kg/m**3",
            val=8920,
            desc="density of pipes",
        )

        # self.add_input(
        #     name="data:thermal:hydrogen:velocity",
        #     units="m/s",
        #     val=,
        #     desc="maximum H2 velocity in gaseous state",
        # )

        # self.add_input(
        #     name="data:thermal:air:velocity",
        #     units="m/s",
        #     val=,
        #     desc="maximum air velocity in gaseous state",
        # )

        self.add_output(name="data:thermal:pipes:coolant:mass", units="kg")

        self.add_output(name="data:thermal:pipes:radius")
        # self.add_output(name="data:thermal:pipes:hydrogen:mass", units="kg")
        # self.add_output(name="data:thermal:pipes:air:mass", units="kg")
        # self.add_output(name="data:thermal:pipes:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_flow = inputs["data:thermal:coolant:mass_flow"]
        rho_coolant = inputs["data:thermal:coolant:density"]
        v_cool = inputs["data:thermal:coolant:velocity"]
        d_pipe = inputs["data:thermal:pipes:distance"]
        t_pipe = inputs["data:thermal:pipes:thickness"]
        rho_pipe = inputs["data:thermal:pipes:density"]

        R_pipe = np.sqrt(m_flow / (np.pi * rho_coolant * v_cool))

        M_coolant_pipe = rho_pipe * np.pi * d_pipe * [(R_pipe + t_pipe) ** 2 - R_pipe**2]
        # M_H2_pipe =
        # M_air_pipe =
        # M_total = M_coolant_pipe + M_H2_pipe + M_air_pipe

        outputs["data:thermal:pipes:radius"] = R_pipe
        outputs["data:thermal:pipes:coolant:mass"] = M_coolant_pipe
        # outputs["data:thermal:pipes:mass"] = M_total

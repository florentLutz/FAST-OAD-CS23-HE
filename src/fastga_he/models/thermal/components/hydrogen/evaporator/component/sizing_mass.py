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

        self.add_input(
            name="data:thermal:hydrogen:evaporator:material_strength",
            units="Pa",
            val=8.07*1e8,
            desc="material ultimate tensile strength",
        )

        self.add_output(name="data:thermal:hydrogen:evaporator:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        t =
        M_H2_evaporator = 
        outputs["data:thermal:hydrogen:evaporator:mass"] = M_H2_evaporator

import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeEvaporatorVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:evaporator:width",
            units="m",
            val=np.nan,
            desc="evaporator width",
        )

        self.add_input(
            name="data:thermal:evaporator:height",
            units="m",
            val=np.nan,
            desc="evaporator height",
        )

        self.add_input(
            name="data:thermal:evaporator:length",
            units="m",
            val=np.nan,
            desc="evaporator length",
        )

        self.add_output(
            name="data:thermal:evaporator:volume",
            units="m**3"
            desc="HEX volume"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        width = inputs["data:thermal:evaporator:width"]
        height = inputs["data:thermal:evaporator:height"]
        length = inputs["data:thermal:evaporator:length"]

        volume = width*height*length
        outputs["data:thermal:evaporator:volume"] = volume

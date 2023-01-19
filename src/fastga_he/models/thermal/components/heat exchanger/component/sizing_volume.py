import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeHEXVolume(om.ExplicitComponent):
    def setup(self):

        self.add_input(
            name="data:thermal:HEX:width",
            units="m",
            val=np.nan,
            desc="HEX width",
        )

        self.add_input(
            name="data:thermal:HEX:height",
            units="m",
            val=np.nan,
            desc="HEX height",
        )

        self.add_input(
            name="data:thermal:HEX:length",
            units="m",
            val=np.nan,
            desc="HEX length",
        )

        self.add_output(
            name="data:thermal:HEX:volume",
            units="m**3"
            desc="HEX volume"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        width = inputs["data:thermal:HEX:width"]
        height = inputs["data:thermal:HEX:height"]
        length = inputs["data:thermal:HEX:length"]

        volume = width*height*length
        outputs["data:thermal:HEX:volume"] = volume

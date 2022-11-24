import numpy as np
import scipy.constants as sc
import openmdao.api as om

class ComputeHydrogenTank(om.ExplicitComponent):
    
    # Volume model from assumptions in: 
    # Palladino, V., Jordan, A., Bartoli, N., Schmollgruber, P., Pommier-Budinger, V., and Benard, E., 
    # “Preliminary studies of aregional aircraft with hydrogen-based hybrid propulsion,” AIAA Aviation, 2021
    
    def setup(self):

        self.add_input(name="H2_mass_flow", units = "kg/s", val=np.nan)
        self.add_input(name="H2_gravimetric_efficiency", val=np.nan)
        self.add_input(name="H2_volumetric_efficiency", val=np.nan)
        self.add_input(name="time", units="s", val=np.nan)

        self.add_output(name="H2_tank_mass", units="kg")
        self.add_output(name="H2_tank_volume", units="m**3")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        M_H2 = inputs["H2_mass_flow"]*inputs["time"]
        n_g = inputs["H2_gravimetric_efficiency"]
        n_v = inputs["H2_volumetric_efficiency"]

        M_H2_tank = M_H2/n_g - M_H2
        V_H2_tank = 0.0366*M_H2 + 0.1215 

        outputs["H2_tank_mass"] = M_H2_tank
        outputs["H2_tank_volume"] = V_H2_tank

import numpy as np
import scipy.constants as sc
import openmdao.api as om


class ComputeCompressorPower(om.ExplicitComponent):
    """
    Sizes the compressor power
    """

    def setup(self):

        ## Inputs

        self.add_input(name="settings:thermal:air:specific_gases_ratio", val=1.4)

        # Compressor
        self.add_input(name="data:thermal:compressor:efficiency", val=np.nan)

        # Fuel cell
        self.add_input(name="data:thermal:fuel_cell_stack:pressure", units="Pa", val=np.nan)

        # Air
        self.add_input(name="data:thermal:fuel_cell_stack:air_mass_flow", units="kg/s", val=np.nan)
        self.add_input(
            name="data:thermal:compressor:air:specific_heat_capacity", units="J/kg/K", val=np.nan
        )
        self.add_input(name="settings:thermal:atmospheric_air:temperature", units="K", val=np.nan)
        self.add_input(
            name="settings:thermal:atmospheric_air:dynamic_pressure", units="Pa", val=np.nan
        )

        ## Outputs
        self.add_output(name="data:thermal:compressor:power", units="W")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Assigning the input to local variable for clarity
        gamma = inputs["settings:thermal:air:specific_gases_ratio"]

        Pr_fc = inputs["data:thermal:fuel_cell_stack:pressure"]

        n = inputs["data:thermal:compressor:efficiency"]

        m_flow = inputs["data:thermal:fuel_cell_stack:air_mass_flow"]
        cp = inputs["data:thermal:compressor:air:specific_heat_capacity"]
        T_air = inputs["settings:thermal:atmospheric_air:temperature"]
        Pr_air = inputs["settings:thermal:atmospheric_air:dynamic_pressure"]

        P = cp * T_air / n * ((Pr_fc / Pr_air) ** ((gamma - 1) / (gamma)) - 1) * m_flow

        outputs["data:thermal:compressor:power"] = P

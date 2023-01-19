import numpy as np
import scipy.constants as sc
import openmdao.api as om
from fluid_characteristics import FluidCharacteristics
from scipy.integrate import odeint


class ComputeCoolantTankRate(om.ExplicitComponent):
    """
    Describing coolant tank

    coolant list and numbering:
    air: 1
    water: 2
    hydrogen: 3
    ammonia: 4
    ethylene glycol: 5
    propylene glycol: 6
    potassium formate: 7
    R134a: 8

    """

    def setup(self):

        self.add_input(
            name="data:thermal:coolant:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="coolant tank rate",
        )

        self.add_input(
            name="data:thermal:coolant:mass",
            units="kg",
            val=np.nan,
            desc="coolant mass with extra factor",
        )

        self.add_input(
            name="data:thermal:coolant:entry_temperature",
            units="K",
            val=np.nan,
            desc="entry temperature of coolant",
        )

        self.add_input(
            name="data:thermal:coolant:entry_pressure",
            units="Pa",
            val=np.nan,
            desc="entry pressure of coolant",
        )

        self.add_input(
            name="data:thermal:coolant:specific_heat_capacity",
            units="J/kg/K",
            val=np.nan,
            desc="coolant specific heat capacity",
        )

        self.add_input(
            name="data:thermal:coolant:type",
            val=1,
            desc="type of coolant",
        )

        self.add_input(
            name="data:time",
            units="s",
            val=np.nan,
            desc="mission time",
        )
        
        self.add_output(name="data:thermal:pipes:coolant_tank:exit_temperature", units="K")
        self.add_output(name="data:thermal:pipes:coolant_tank:rate", units="K/s")

        # self.declare_partials(
        #     of="data:thermal:pipes:coolant_tank:rate",
        #     wrt="data:thermal:coolant:mass_flow",
        #     method="exact",
        # )

        # self.declare_partials(
        #     of="data:thermal:pipes:coolant_tank:rate",
        #     wrt="data:thermal:coolant:mass",
        #     method="exact",
        # ) 

        # self.declare_partials(
        #     of="data:thermal:pipes:coolant_tank:rate",
        #     wrt="data:thermal:coolant_tank:entry_temperature",
        #     method="exact",
        # )

        # self.declare_partials(
        #     of="data:thermal:pipes:coolant_tank:rate",
        #     wrt="data:thermal:coolant_tank:exit_temperature",
        #     method="exact",
        # )       

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None, partials):

        m_cool = inputs["data:thermal:coolant:mass_flow"]
        M_cool = inputs["data:thermal:coolant:mass"]
        T_cool_in = inputs["data:thermal:coolant_tank:entry_temperature"]
        P_cool = inputs["data:thermal:coolant_tank:entry_pressure"]
        cp_cool = inputs["data:thermal:coolant:specific_heat_capacity"]
        cool_type = inputs["data:thermal:coolant:type"]
        t =  inputs["data:time"]

        if coolant_type == 1:
            coolant_type = "air"
        elif coolant_type == 2:
            coolant_type = "water"
        elif coolant_type == 3:
            coolant_type = "hydrogen"
        elif coolant_type == 4:
            coolant_type = "ammonia"
        elif coolant_type == 5:
            coolant_type = "ethylene glycol"
        elif coolant_type == 6:
            coolant_type = "propylene glycol"
        elif coolant_type == 7:
            coolant_type = "potassium formate"
        elif coolant_type == 8:
            coolant_type = "R134a"

        coolant = FluidCharacteristics(
            temperature=T_cool_in,
            pressure=P_cool,
            coolant=cool_type,
        )

        cp_cool = coolant.specific_heat_capacity

        def compute_temperature(T_initial, T_cool_in, M_cool, m_cool, dTdt_func, t, num_steps):
            T_cool_out = T_cool_in - 5
            for i in range(num_steps):
                dTdt = dTdt_func(m_cool, M_cool, T_cool_out, T_cool_in)
                T_cool_out += dTdt*t
            return T_cool_out

        def dTdt_func(m_cool, M_cool, T_cool_out, T_cool_in):
            return m_cool / M_cool * (T_cool_in - T_cool_out)

        T_cool_out = compute_temperature(300, 310, 100, 0.1, dTdt_func, 0.1, 100)
        dTdt = dTdt_func(m_cool, M_cool, T_cool_out, T_cool_in)

        # partials["data:thermal:pipes:coolant_tank:rate", "data:thermal:coolant:M_cool"] = -m_cool / (M_cool ** 2) * (T_cool_in - T_cool_out)
        # partials["data:thermal:pipes:coolant_tank:rate", "data:thermal:coolant:M_cool_flow"] = 1 / M_cool * (T_cool_in - T_cool_out)
        # partials["data:thermal:pipes:coolant_tank:rate", "data:thermal:coolant_tank:entry_temperature"] = m_cool / M_cool
        # partials["data:thermal:pipes:coolant_tank:rate", "data:thermal:coolant_tank:exit_temperature"] = - m_cool / M_cool

        outputs["data:thermal:coolant_tank:exit_temperature"] = T_cool_out
        outputs["data:thermal:pipes:coolant_tank:rate"] = dTdt

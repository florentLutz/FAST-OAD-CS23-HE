import numpy as np
import scipy.constants as sc
import openmdao.api as om
from utils.fluid_characteristics import FluidCharacteristics
from scipy.optimize import root
from scipy.optimize import minimize, fsolve

class ComputePerformanceEvaporator(om.ExplicitComponent):
    """
    Evaporator model
    """
    """
    fluids list and numbering:
    air: 1
    water: 2
    hydrogen: 3
    ammonia: 4
    ethylene glycol: 5
    propylene glycol: 6
    potassium formate: 7
    R134a: 8
    liquid hydrogen: 9
    """

def setup(self):

        ##  general evaporator characteristics
        self.add_input(
            name="data:thermal:evaporator:effectiveness",
            val=0.8,
            desc="evaporator effectivenes",
        )
        self.add_input(
            name="settings:thermal:evaporator:gas_side:hydraulic_diameter",
            units="m",
            val=0.0036,
            desc="hydraulic diameter of gas side",
        )
        self.add_input(
            name="settings:thermal:evaporator:gas_side:surface_area_to_volume_ratio",
            units="m**(-1)",
            val=751,
            desc="ratio of evaporator surface area to volume ratio on the gas side",
        )
        self.add_input(
            name="settings:thermal:evaporator:gas_side:flow_surface_to_frontal_surface_area_ratio",
            val=0.697,
            desc="ratio of evaporator flow surface to frontal surface area on the gas side",
        )
        self.add_input(
            name="settings:thermal:evaporator:fin_surface_to_total_surface_area_ratio",
            val=0.795,
            desc="evaporator fin surface to total heat exchanger surface area",
        )
        self.add_input(
            name="settings:thermal:evaporator:fin:thickness",
            units="m",
            val=0.0001,
            desc="evaporator fin thickness",
        )
        self.add_input(
            name="settings:thermal:evaporator:fin:half_length",
            units="m",
            val=0.004011,
            desc="evaporator final half length",
        )
        self.add_input(
            name="settings:thermal:evaporator:fin:thermal_conductivity",
            units="W/m/K",
            val=200,
            desc="evaporator fin thermal conductivity",
        )
        self.add_input(
            name="settings:thermal:evaporator:liquid_side:hydraulic_diameter",
            units="m",
            val=5.48e-03,
            desc="hydraulic diameter of liquid side",
        )
        self.add_input(
            name="settings:thermal:evaporator:liquid_side:surface_area_to_volume_ratio",
            units="m**(-1)",
            val=1.60e2,
            desc="ratio of evaporator surface area to volume ratio on the liquid side",
        )
        self.add_input(
            name="settings:thermal:evaporator:liquid_side:flow_surface_to_frontal_surface_area_ratio",
            val=2.19e-01,
            desc="ratio of evaporator flow surface to frontal surface area on the liquid side",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:frontal_surface",
            units="m**2",
            val=0.000298,
            desc="Frontal surface area of evaporator",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:flow_section",
            units="m**2",
            val=6.54e-05,
            desc="heat exchanger ",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:height",
            units="m",
            val=0.00305,
            desc="evaporator tube height",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:length",
            units="m",
            val=0.0221,
            desc="evaporator tube length",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:thickness",
            units="m",
            val=0.0001,
            desc="evaporator tube thickness",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:vertical_separation",
            units="m",
            val=0.00802,
            desc="evaporator tube vertical separation",
        )
        self.add_input(
            name="settings:thermal:evaporator:tube:horizontal_separation",
            units="m",
            val=0.00482,
            desc="evaporator tube horizontal separation",
        )
        self.add_input(
            name="settings:thermal:evaporator:plate:separation",
            units="m",
            val=0.00252,
            desc="evaporator plate separation",
        )
        self.add_input(
            name="settings:thermal:evaporator:material_density",
            units="kg/m**3",
            val=2712,
            desc="evaporator material density",
        )

        ## fluid characteristics
        self.add_input(
            name="data:thermal:gas_fluid:type",
            val=1,
            desc="type of gas fluid",
        )
        self.add_input(
            name="data:thermal:gas_fluid:entry_temperature",
            units="K",
            val=np.nan,
            desc="gas fluid entry temperature",
        )
        self.add_input(
            name="data:thermal:gas_fluid:exit_temperature",
            units="K",
            val=np.nan,
            desc="gas fluid exit temperature",
        )
        self.add_input(
            name="data:thermal:gas_fluid:pressure",
            units="Pa",
            val=np.nan,
            desc="gas fluid pressure",
        )
        self.add_input(
            name="data:thermal:gas_fluid:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="gas fluid mass flow",
        )
        self.add_input(
            name="data:thermal:liquid_fluid:type",
            val=9,
            desc="type of liquid fluid",
        )
        self.add_input(
            name="data:thermal:liquid_fluid:entry_temperature",
            units="K",
            val=np.nan,
            desc="liquid fluid entry temperature",
        )
        self.add_input(
            name="data:thermal:liquid_fluid:exit_temperature",
            units="K",
            val=np.nan,
            desc="liquid fluid exit temperature",
        )
        self.add_input(
            name="data:thermal:liquid_fluid:pressure",
            units="Pa",
            val=np.nan,
            desc="liquid fluid pressure",
        )
        self.add_input(
            name="data:thermal:liquid_fluid:mass_flow",
            units="kg/s",
            val=np.nan,
            desc="liquid fluid mass flow",
        )

        ## outputs
        self.add_output(
            name="data:thermal:evaporator:NTU",
            desc="evaporator number of transfer units",
        )
        self.add_output(
            name="data:thermal:evaporator:UA",
            desc="evaporator UA",
        )
        self.add_output(
            name="data:thermal:evaporator:width",
            units="m",
            desc="evaporator width",
        )
        self.add_output(
            name="data:thermal:evaporator:height",
            units="m",
            desc="evaporator height",
        )
        self.add_output(
            name="data:thermal:evaporator:length",
            units="m",
            desc="evaporator length",
        )
        self.add_output(
            name="data:thermal:evaporator:unit_density",
            units="kg/m**3",
            desc="evaporator unit density",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        ## general evaporator characteristics
        D_h_gas = inputs["settings:thermal:evaporator:gas_side:hydraulic_diameter"]
        alpha_gas = inputs["settings:thermal:evaporator:gas_side:surface_area_to_volume_ratio"]
        sigma_gas = inputs[
            "settings:thermal:evaporator:gas_side:flow_surface_to_frontal_surface_area_ratio"
        ]
        t_fin = inputs["settings:thermal:evaporator:fin:thickness"]
        half_fin = inputs["settings:thermal:evaporator:fin:half_length"]
        lambda_fin = inputs["settings:thermal:evaporator:fin:thermal_conductivity"]
        fin_surface_to_total_surface_ratio = inputs[
            "settings:thermal:evaporator:fin_surface_to_total_surface_area_ratio"
        ]

        D_h_liquid = inputs["settings:thermal:evaporator:liquid_side:hydraulic_diameter"]
        alpha_liquid = inputs["settings:thermal:evaporator:liquid_side:surface_area_to_volume_ratio"]
        sigma_liquid = inputs[
            "settings:thermal:evaporator:liquid_side:flow_surface_to_frontal_surface_area_ratio"
        ]
        A_flow_liquid = inputs["settings:thermal:evaporator:tube:flow_section"]
        t_tube = t_fin  # assuming tube thickness same as fin thickness

        tube_height = inputs["settings:thermal:evaporator:tube:height"]
        tube_length = inputs["settings:thermal:evaporator:tube:length"]
        tube_vsep = inputs["settings:thermal:evaporator:tube:vertical_separation"]
        tube_hsep = inputs["settings:thermal:evaporator:tube:horizontal_separation"]
        plate_sep = inputs["settings:thermal:evaporator:plate:separation"]

        rho_material = inputs["settings:thermal:evaporator:material_density"]

        ## gas fluid basic
        gas_type = inputs["data:thermal:gas_fluid:type"]
        T_gas_in = inputs["data:thermal:gas_fluid:entry_temperature"]
        T_gas_out = inputs["data:thermal:gas_fluid:exit_temperature"]
        P_gas = inputs["data:thermal:gas_fluid:pressure"]
        m_gas = inputs["data:thermal:gas_fluid:mass_flow"]
        T_gas_avg = (T_gas_out + T_gas_in) / 2
        Delta_T_gas = T_gas_out - T_gas_in

        if gas_type == 1:
            gas_type = "air"
        elif gas_type == 2:
            gas_type = "water"
        elif gas_type == 3:
            gas_type = "hydrogen"
        elif gas_type == 4:
            gas_type = "ammonia"
        elif gas_type == 5:
            gas_type = "ethylene glycol"
        elif gas_type == 6:
            gas_type = "propylene glycol"
        elif gas_type == 7:
            gas_type = "potassium formate"
        elif gas_type == 8:
            gas_type = "R134a"
        elif gas_type == 9:
            gas_type = "liquid hydrogen"

        gas_fluid = FluidCharacteristics(
            temperature=T_gas_avg,
            pressure=P_gas,
            coolant=gas_type,
        )
        rho_gas = gas_fluid.density
        cp_gas = gas_fluid.specific_heat_capacity
        mu_gas = gas_fluid.dynamic_viscosity
        lambda_gas = gas_fluid.thermal_conductivity
        Pr_gas = gas_fluid.Prandtl
        nu_gas = gas_fluid.specific_volume
        C_gas = m_gas * cp_gas

        ## liquid fluid basic
        liquid_type = inputs["data:thermal:liquid_fluid:type"]
        T_liquid_in = inputs["data:thermal:liquid_fluid:entry_temperature"]
        T_liquid_out = inputs["data:thermal:liquid_fluid:exit_temperature"]
        P_liquid = inputs["data:thermal:liquid_fluid:pressure"]
        m_liquid = inputs["data:thermal:liquid_fluid:mass_flow"]
        T_liquid_avg = (T_liquid_out + T_liquid_in) / 2
        Delta_T_liquid = T_liquid_out - T_liquid_in

        if liquid_type == 1:
            liquid_type = "air"
        elif liquid_type == 2:
            liquid_type = "water"
        elif liquid_type == 3:
            liquid_type = "hydrogen"
        elif liquid_type == 4:
            liquid_type = "ammonia"
        elif liquid_type == 5:
            liquid_type = "ethylene glycol"
        elif liquid_type == 6:
            liquid_type = "propylene glycol"
        elif liquid_type == 7:
            liquid_type = "potassium formate"
        elif liquid_type == 8:
            liquid_type = "R134a"
        elif liquid_type == 9:
            liquid_type = "liquid hydrogen"

        liquid_fluid = FluidCharacteristics(
            temperature=T_liquid_avg,
            pressure=P_liquid,
            coolant=liquid_type,
        )
        rho_liquid = liquid_fluid.density
        cp_liquid = liquid_fluid.specific_heat_capacity
        mu_liquid = liquid_fluid.dynamic_viscosity
        lambda_liquid = liquid_fluid.thermal_conductivity
        Pr_liquid = liquid_fluid.Prandtl
        nu_liquid = liquid_fluid.specific_volume

        ## general
        eff = inputs["data:thermal:evaporator:effectiveness"]

        func = lambda x: eff - (
            1 - np.e ** (-x[0])
        )
        res = root(func, x0=1)
        [NTU] = res.x

        UA = NTU * C_gas # need UA gasing requirements to match UA given by evaporator dimensions

        def objective(x):
            length, height, width = x
            global UA_new

            frontal_surface_area_gas = width * height
            frontal_surface_area_liquid = length * height

            v_gas = m_gas / (rho_gas * sigma_gas * frontal_surface_area_gas)
            Re_gas = m_gas * D_h_gas / (mu_gas * sigma_gas * frontal_surface_area_gas)
            Colburn_gas = (0.5615 * Re_gas ** (-0.653)) ** (1 / (1 + (Re_gas / 2000) ** 9)) * (
                0.0225 * Re_gas ** (-0.22)
            ) ** (1 - (1 / (1 + (Re_gas / 2000) ** 9)))

            if Re_gas < 2000:
                Darcy_gas = (64 / Re_gas) ** (1 / (1 + (Re_gas / 2720) ** 9)) * (
                    1.8 * np.log10(Re_gas / 6.8)
                ) ** (-2 * (1 - (1 / (1 + (Re_gas / 2720) ** 9))))
            else:
                Darcy_gas = (2.7 * Re_gas ** (-0.68)) ** (1 / (1 + (Re_gas / 2000) ** 9)) * (
                    0.13 * Re_gas ** (-0.28)
                ) ** (1 - (1 / (1 + (Re_gas / 2000) ** 9)))

            St_gas = Colburn_gas / Pr_gas ** (2 / 3)
            h_gas = St_gas * rho_gas * v_gas * cp_gas

            S_gas = alpha_gas * length * height * width
            no_gas = 1 - fin_surface_to_total_surface_ratio * (
                1
                - np.tanh(np.sqrt(2 * h_gas / t_fin / lambda_fin) * half_fin)
                / (np.sqrt(2 * h_gas / t_fin / lambda_fin) * half_fin)
            )

            v_liquid = m_liquid / (rho_liquid * sigma_liquid * frontal_surface_area_liquid)
            Re_liquid = (
                m_liquid * D_h_liquid / (mu_liquid * sigma_liquid * frontal_surface_area_liquid)
            )
            Colburn_liquid = (0.5615 * Re_liquid ** (-0.653)) ** (
                1 / (1 + (Re_liquid / 2000) ** 9)
            ) * (0.0225 * Re_liquid ** (-0.22)) ** (1 - (1 / (1 + (Re_liquid / 2000) ** 9)))

            if Re_liquid < 2000:
                Darcy_liquid = (64 / Re_liquid) ** (1 / (1 + (Re_liquid / 2720) ** 9)) * (
                    1.8 * np.log10(Re_liquid / 6.8)
                ) ** (-2 * (1 - (1 / (1 + (Re_liquid / 2720) ** 9))))
            else:
                Darcy_liquid = (2.7 * Re_liquid ** (-0.68)) ** (
                    1 / (1 + (Re_liquid / 2000) ** 9)
                ) * (0.13 * Re_liquid ** (-0.28)) ** (1 - (1 / (1 + (Re_liquid / 2000) ** 9)))

            if Re_liquid > 2000:
                Nu_liquid = (
                    (Darcy_liquid / 8)
                    * (Re_liquid - 1000)
                    * Pr_liquid
                    / (1 + 12.7 * np.sqrt(Darcy_liquid / 8) * (Pr_liquid ** (2 / 3) - 1))
                )
            else:
                Nu_liquid = 3.66

            h_liquid = Nu_liquid * lambda_liquid / D_h_liquid
            S_liquid = alpha_liquid * length * height * width
            no_liquid = 1

            UA_new = 1 / (1 / (h_gas * S_gas * no_gas) + 1 / (h_liquid * S_liquid * no_liquid))

            return abs(UA_new - UA)

        initial_guesses = [(0.1, 0.1, 0.1), (0.7, 0.5, 0.1), (1, 0.7, 0.1)]
        bounds = [(tube_length, None), (0.01107, None), (plate_sep + t_fin, None)]

        results = []

        for x0 in initial_guesses:
            res = minimize(objective, x0, bounds=bounds)
            results.append(res)

        best_result = min(results, key=lambda x: x.fun)
        
        if best_result.fun < 1e-5
            # print(f"length: {best_result.x[0]}, height: {best_result.x[1]}, width: {best_result.x[2]}")
            # print(best_result)
            length = best_result.x[0]
            height = best_result.x[1]
            width = best_result.x[2]

        M_unit_plate = (
            ((tube_length + tube_hsep) * (tube_vsep + tube_height) - tube_height * tube_length)
            * t_fin
            * rho_material
        )
        M_unit_cool = rho_liquid * (plate_sep + t_fin) * A_flow_liquid
        M_unit_tube = (
            (
                (tube_height * tube_length)
                - ((tube_height - 2 * t_tube) * (tube_length - 2 * t_tube))
            )
            * (plate_sep + t_fin)
            * rho_material
        )
        M_unit_evaporator = M_unit_plate + M_unit_cool + M_unit_tube
        rho_unit_evaporator = M_unit_evaporator / (
            (plate_sep + t_fin) * (tube_vsep + tube_height) * (tube_length + tube_hsep)
        )

        outputs["data:thermal:evaporator:NTU"] = NTU
        outputs["data:thermal:evaporator:UA"] = UA
        outputs["data:thermal:evaporator:length"] = length
        outputs["data:thermal:evaporator:height"] = height
        outputs["data:thermal:evaporator:width"] = width
        outputs["data:thermal:evaporator:unit_density"] = rho_unit_evaporator
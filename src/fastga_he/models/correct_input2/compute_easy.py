# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain
from stdatm import AtmosphereWithPartials


@oad.RegisterOpenMDAOSystem("fastga_he.correct_input.compute_easy2", domain=ModelDomain.GEOMETRY)
class EASY_compute2(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)
        self.add_input("data:TLAR:cruise_mach", val=np.nan)
        self.add_input("data:TLAR:approach_speed", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:taxi_in:distance", val=np.nan, units="m")
        self.add_input("data:mission:sizing:taxi_in:duration", val=np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_out:distance", val=np.nan, units="m")
        self.add_input("data:mission:sizing:taxi_out:duration", val=np.nan, units="s")
        self.add_input("data:propulsion:L1_engine:hpc:hpc_pressure_ratio", val=np.nan)
        self.add_input("data:propulsion:L1_engine:lpc:lpc_pressure_ratio", val=np.nan)
        self.add_input("data:propulsion:Design_Thermo_Power", val=np.nan, units="W")
        self.add_input("data:propulsion:RTO_power", val=np.nan, units="W")

        self.add_output("data:TLAR:luggage_mass_design", units="kg")
        self.add_output('data:aerodynamics:cruise:unit_reynolds', units="1/m")
        self.add_output('data:aerodynamics:low_speed:unit_reynolds', units="1/m")
        self.add_output("data:TLAR:v_cruise", units="m/s")
        self.add_output("data:mission:sizing:taxi_in:speed", units="m/s")
        self.add_output("data:mission:sizing:taxi_out:speed", units="m/s")
        # turboshaft
        """self.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:OPR")
        self.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_1:limit:OPR")
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:power_ratio"
        )

        self.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:OPR")
        self.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_2:limit:OPR")
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:power_ratio"
        )"""

        self.declare_partials(
            of="data:TLAR:luggage_mass_design", wrt="data:TLAR:NPAX_design", val=20
        )
        self.declare_partials(
            of="data:TLAR:v_cruise", wrt="data:mission:sizing:main_route:cruise:altitude"
        )

        self.declare_partials(
            of="data:mission:sizing:taxi_in:speed", wrt="data:mission:sizing:taxi_in:distance"
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_in:speed", wrt="data:mission:sizing:taxi_in:duration"
        )

        self.declare_partials(
            of="data:mission:sizing:taxi_out:speed", wrt="data:mission:sizing:taxi_out:distance"
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_out:speed", wrt="data:mission:sizing:taxi_out:duration"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        atm = AtmosphereWithPartials(cruise_alt, altitude_in_feet=False)
        sos = atm.speed_of_sound

        V_TAS = sos * inputs["data:TLAR:cruise_mach"]
        atm.true_airspeed = V_TAS
        unit_Reynolds = atm.unitary_reynolds

        low_speed_alt = 0
        atm2 = AtmosphereWithPartials(low_speed_alt, altitude_in_feet=False)
        atm2.true_airspeed = inputs["data:TLAR:approach_speed"]
        unit_Reynolds2 = atm2.unitary_reynolds

        outputs["data:TLAR:v_cruise"] = V_TAS

        outputs['data:aerodynamics:cruise:unit_reynolds'] = unit_Reynolds

        outputs['data:aerodynamics:low_speed:unit_reynolds'] = unit_Reynolds2

        outputs["data:TLAR:luggage_mass_design"] = 20 * inputs["data:TLAR:NPAX_design"]

        outputs["data:mission:sizing:taxi_in:speed"] = (
            inputs["data:mission:sizing:taxi_in:distance"]
            / inputs["data:mission:sizing:taxi_in:duration"]
        )

        outputs["data:mission:sizing:taxi_out:speed"] = (
            inputs["data:mission:sizing:taxi_out:distance"]
            / inputs["data:mission:sizing:taxi_out:duration"]
        )

        # turboshaft
        """outputs["data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:OPR"] = (
            inputs["data:propulsion:L1_engine:hpc:hpc_pressure_ratio"]
            * inputs["data:propulsion:L1_engine:lpc:lpc_pressure_ratio"]
        )

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_1:limit:OPR"] = (
            inputs["data:propulsion:L1_engine:hpc:hpc_pressure_ratio"]
            * inputs["data:propulsion:L1_engine:lpc:lpc_pressure_ratio"]
        )

        outputs[
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:power_ratio"
        ] = inputs["data:propulsion:Design_Thermo_Power"] / inputs["data:propulsion:RTO_power"]

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:OPR"] = (
            inputs["data:propulsion:L1_engine:hpc:hpc_pressure_ratio"]
            * inputs["data:propulsion:L1_engine:lpc:lpc_pressure_ratio"]
        )

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_2:limit:OPR"] = (
            inputs["data:propulsion:L1_engine:hpc:hpc_pressure_ratio"]
            * inputs["data:propulsion:L1_engine:lpc:lpc_pressure_ratio"]
        )

        outputs[
            "data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:power_ratio"
        ] = inputs["data:propulsion:Design_Thermo_Power"] / inputs["data:propulsion:RTO_power"]"""

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        # cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        # sos = AtmosphereWithPartials(cruise_alt, altitude_in_feet=False).speed_of_sound
        # partials["tip_mach", "altitude"] =  inputs["data:TLAR:cruise_mach"]*sos

        partials["data:mission:sizing:taxi_in:speed", "data:mission:sizing:taxi_in:distance"] = (
            1.0 / inputs["data:mission:sizing:taxi_in:duration"]
        )
        partials["data:mission:sizing:taxi_in:speed", "data:mission:sizing:taxi_in:duration"] = -(
            inputs["data:mission:sizing:taxi_in:distance"]
        ) / (inputs["data:mission:sizing:taxi_in:duration"] ** 2.0)

        partials["data:mission:sizing:taxi_out:speed", "data:mission:sizing:taxi_out:distance"] = (
            1.0 / inputs["data:mission:sizing:taxi_in:duration"]
        )
        partials["data:mission:sizing:taxi_out:speed", "data:mission:sizing:taxi_out:duration"] = -(
            inputs["data:mission:sizing:taxi_out:distance"]
        ) / (inputs["data:mission:sizing:taxi_out:duration"] ** 2.0)

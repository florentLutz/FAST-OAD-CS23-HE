# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem("fastga_he.correct_input.set_name")
class SetValue(om.ExplicitComponent):
    """ Some variables with standard values are required in HE as input,
     but they are not required as input in RTA, or some values are to complex to be computed"""
    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)  # not used

        # standard value
        self.add_output("data:mission:sizing:takeoff:energy", units="W*h")
        self.add_output("data:mission:sizing:initial_climb:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:reserve:duration", units="s")
        self.add_output("data:geometry:cabin:seats:pilot:length", units="m")
        self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
        self.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="1/rad")

        # performance required as input
        self.add_output("data:mission:sizing:main_route:climb:climb_rate:cruise_level", units="m/s")
        self.add_output("data:mission:sizing:main_route:climb:climb_rate:sea_level", units="m/s")
        self.add_output("data:mission:sizing:main_route:descent:descent_rate", units="m/s")

        # elevator info required as input
        self.add_output("data:geometry:horizontal_tail:elevator_chord_ratio")
        self.add_output("data:mission:sizing:landing:elevator_angle", units="rad")

        # more airfoil input required: dy, maximum camber and its position, position of maximum thickness
        self.add_output("data:aerodynamics:wing:cruise:CM0_clean")
        # the following have been setted considering aircraft parametres
        # self.add_output('data:aerodynamics:horizontal_tail:airfoil:CL_alpha', units="1/rad")
        # self.add_output("data:aerodynamics:wing:cruise:CL0_clean")
        # self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")

        # propeller
        # standard
        self.add_output("data:propulsion:he_power_train:propeller:propeller_1:cl_clean_ref")
        self.add_output("data:propulsion:he_power_train:propeller:propeller_1:flapped_ratio")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_1:installation_angle", units="rad"
        )
        # required as input
        self.add_output("data:propulsion:he_power_train:propeller:propeller_1:activity_factor")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_1:blade_twist", units="rad"
        )
        self.add_output("data:propulsion:he_power_train:propeller:propeller_1:from_wing_LE_ratio")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_1:rpm_mission", units="1/min"
        )
        self.add_output("data:propulsion:he_power_train:propeller:propeller_1:solidity")

        self.add_output("data:propulsion:he_power_train:propeller:propeller_2:activity_factor")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_2:blade_twist", units="rad"
        )
        self.add_output("data:propulsion:he_power_train:propeller:propeller_2:cl_clean_ref")
        self.add_output("data:propulsion:he_power_train:propeller:propeller_2:flapped_ratio")
        self.add_output("data:propulsion:he_power_train:propeller:propeller_2:from_wing_LE_ratio")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_2:installation_angle", units="rad"
        )
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_2:rpm_mission", units="1/min"
        )
        self.add_output("data:propulsion:he_power_train:propeller:propeller_2:solidity")

        # turboshaft from Type Certificate
        self.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_1:limit:ITT")
        self.add_output("data:propulsion:he_power_train:turboshaft:turboshaft_2:limit:ITT")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:mission:sizing:takeoff:energy"] = 0

        outputs["data:mission:sizing:initial_climb:energy"] = 0

        outputs["data:mission:sizing:main_route:reserve:duration"] = 3500.0

        outputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"] = 4.0

        outputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"] = 6.45

        outputs["data:mission:sizing:main_route:descent:descent_rate"] = -7.62

        outputs["data:geometry:horizontal_tail:elevator_chord_ratio"] = 0.384  # 0.384

        outputs[
            "data:mission:sizing:landing:elevator_angle"
        ] = -0.3363323129985824  # -0.6363323129985824

        outputs["data:geometry:cabin:seats:pilot:length"] = 1.05

        outputs["data:aerodynamics:horizontal_tail:cruise:CL0"] = -0.0068437669175491515

        outputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"] = 6.28

        outputs["data:aerodynamics:wing:cruise:CM0_clean"] = -0.02413516654351498

        # PROPELLER

        outputs["data:propulsion:he_power_train:propeller:propeller_1:activity_factor"] = 10.6

        outputs["data:propulsion:he_power_train:propeller:propeller_1:blade_twist"] = (
            0.61066860685499065
        )

        outputs["data:propulsion:he_power_train:propeller:propeller_1:cl_clean_ref"] = 0.0

        outputs["data:propulsion:he_power_train:propeller:propeller_1:flapped_ratio"] = 0

        outputs["data:propulsion:he_power_train:propeller:propeller_1:from_wing_LE_ratio"] = (
            1.613728620523854
        )

        outputs["data:propulsion:he_power_train:propeller:propeller_1:installation_angle"] = 0.0

        outputs["data:propulsion:he_power_train:propeller:propeller_1:rpm_mission"] = 1100.0

        outputs["data:propulsion:he_power_train:propeller:propeller_1:solidity"] = 0.3

        outputs["data:propulsion:he_power_train:propeller:propeller_2:activity_factor"] = 10.6

        outputs["data:propulsion:he_power_train:propeller:propeller_2:blade_twist"] = (
            0.61066860685499065
        )

        outputs["data:propulsion:he_power_train:propeller:propeller_2:cl_clean_ref"] = 0.0

        outputs["data:propulsion:he_power_train:propeller:propeller_2:flapped_ratio"] = 0

        outputs["data:propulsion:he_power_train:propeller:propeller_2:from_wing_LE_ratio"] = (
            1.613728620523854
        )

        outputs["data:propulsion:he_power_train:propeller:propeller_2:installation_angle"] = 0.0

        outputs["data:propulsion:he_power_train:propeller:propeller_2:rpm_mission"] = 1100.0

        outputs["data:propulsion:he_power_train:propeller:propeller_2:solidity"] = 0.3

        # turboshaft
        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_1:limit:ITT"] = 800

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_2:limit:ITT"] = 800

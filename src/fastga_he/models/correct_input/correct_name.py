# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem("fastga_he.correct_input.correct_name")
class HeNameFromRTAname(om.ExplicitComponent):
    """Same variables but with different name in HE and RTA are renamed"""
    def setup(self):
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:geometry:cabin:seats:economical:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:seats:economical:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:takeoff:mach", val=np.nan)
        self.add_input(
            "data:aerodynamics:aircraft:cruise:CL_alpha", val=np.nan, units="1/rad"
        )  # to correct
        self.add_input("data:weight:fuel_tank:CG:x", val=np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:initial_climb:fuel", units="kg")
        self.add_input("data:aerodynamics:aircraft:cruise:CL0", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CL0", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CL", shape_by_conn=True, val=np.nan)
        self.add_input("data:geometry:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")
        self.add_input(
            "data:propulsion:L1_engine:turbine_inlet_temperature", val=np.nan, units="degK"
        )
        self.add_input("data:propulsion:RTO_power", val=np.nan, units="W")
        # self.add_input("data:aerodynamics:aircraft:cruise:CD0", shape_by_conn=True, val=np.nan)
        # self.add_input("data:aerodynamics:aircraft:low_speed:CD0", shape_by_conn=True, val=np.nan)

        self.add_output("data:TLAR:NPAX_design")
        self.add_output("data:geometry:cabin:seats:passenger:count_by_row")
        self.add_output("data:geometry:cabin:seats:passenger:length", units="m")
        self.add_output("data:geometry:wing:wet_area", units="m**2")
        self.add_output("data:aerodynamics:low_speed:mach")
        # self.add_output('data:aerodynamics:horizontal_tail:airfoil:CL_alpha', units="1/rad")       #doubt
        self.add_output("data:propulsion:he_power_train:fuel_tank:fuel_tank_1:CG:x", units="m")
        self.add_output("data:propulsion:he_power_train:fuel_tank:fuel_tank_2:CG:x", units="m")
        self.add_output("data:mission:sizing:main_route:reserve:altitude", units="m")
        self.add_output("data:mission:sizing:initial_climb:fuel", val=np.nan, units="kg")
        self.add_output("data:aerodynamics:wing:cruise:CL0_clean")  # doubt
        self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")  # doubt
        self.add_output("data:aerodynamics:wing:cruise:CL_alpha")
        self.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient")
        self.add_output("data:aerodynamics:wing:low_speed:induced_drag_coefficient")
        self.add_output("data:aerodynamics:wing:low_speed:CL_max_clean")

        # propeller
        self.add_output("data:propulsion:he_power_train:propeller:propeller_1:diameter", units="m")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_1:wing_chord_ref", units="m"
        )

        self.add_output("data:propulsion:he_power_train:propeller:propeller_2:diameter", units="m")
        self.add_output(
            "data:propulsion:he_power_train:propeller:propeller_2:wing_chord_ref", units="m"
        )

        # turboshaft
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t", units="degK"
        )
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating", units="W"
        )
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_1:shaft_power_rating", units="W"
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:T41t", units="degK"
        )
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_2:power_rating", units="W"
        )
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:turboshaft_2:shaft_power_rating", units="W"
        )

        """#fromArraytoFloat
        self.add_output('data:aerodynamics:aircraft:cruise:CD0')           #doubt
        self.add_output('data:aerodynamics:aircraft:low_speed:CD0')           #doubt
        self.add_output('data:aerodynamics:wing:cruise:CD0')           #doubt
"""

        self.declare_partials(of="data:TLAR:NPAX_design", wrt="data:TLAR:NPAX", val=1.0)
        self.declare_partials(
            of="data:geometry:cabin:seats:passenger:count_by_row",
            wrt="data:geometry:cabin:seats:economical:count_by_row",
            val=1.0,
        )
        self.declare_partials(
            of="data:geometry:cabin:seats:passenger:length",
            wrt="data:geometry:cabin:seats:economical:length",
            val=1.0,
        )
        self.declare_partials(
            of="data:geometry:wing:wet_area", wrt="data:geometry:wing:wetted_area", val=1.0
        )
        self.declare_partials(
            of="data:aerodynamics:low_speed:mach",
            wrt="data:aerodynamics:aircraft:takeoff:mach",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:fuel_tank:fuel_tank_1:CG:x",
            wrt="data:weight:fuel_tank:CG:x",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:fuel_tank:fuel_tank_2:CG:x",
            wrt="data:weight:fuel_tank:CG:x",
            val=1.0,
        )

        self.declare_partials(
            of="data:mission:sizing:main_route:reserve:altitude",
            wrt="data:mission:sizing:main_route:cruise:altitude",
            val=1.0,
        )
        self.declare_partials(
            of="data:mission:sizing:initial_climb:fuel",
            wrt="data:mission:sizing:main_route:initial_climb:fuel",
            val=1.0,
        )

        self.declare_partials(
            of="data:aerodynamics:wing:cruise:CL0_clean",
            wrt="data:aerodynamics:aircraft:cruise:CL0",
            val=1.0,
        )
        self.declare_partials(
            of="data:aerodynamics:wing:low_speed:CL0_clean",
            wrt="data:aerodynamics:aircraft:low_speed:CL0",
            val=1.0,
        )
        self.declare_partials(
            of="data:aerodynamics:wing:cruise:CL_alpha",
            wrt="data:aerodynamics:aircraft:cruise:CL_alpha",
            val=1.0,
        )
        self.declare_partials(
            of="data:aerodynamics:wing:cruise:induced_drag_coefficient",
            wrt="data:aerodynamics:aircraft:cruise:induced_drag_coefficient",
            val=1.0,
        )
        self.declare_partials(
            of="data:aerodynamics:wing:low_speed:induced_drag_coefficient",
            wrt="data:aerodynamics:aircraft:low_speed:induced_drag_coefficient",
            val=1.0,
        )

        self.declare_partials(
            of="data:aerodynamics:wing:low_speed:CL_max_clean",
            wrt="data:aerodynamics:aircraft:low_speed:CL",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:propeller_1:diameter",
            wrt="data:geometry:propulsion:propeller:diameter",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:propeller_1:wing_chord_ref",
            wrt="data:geometry:wing:kink:chord",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:propeller_2:diameter",
            wrt="data:geometry:wing:kink:chord",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:propeller:propeller_2:wing_chord_ref",
            wrt="data:geometry:wing:kink:chord",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t",
            wrt="data:propulsion:L1_engine:turbine_inlet_temperature",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating",
            wrt="data:propulsion:RTO_power",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:turboshaft_1:shaft_power_rating",
            wrt="data:propulsion:RTO_power",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:T41t",
            wrt="data:propulsion:L1_engine:turbine_inlet_temperature",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:turboshaft_2:power_rating",
            wrt="data:propulsion:RTO_power",
            val=1.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:turboshaft_2:shaft_power_rating",
            wrt="data:propulsion:RTO_power",
            val=1.0,
        )
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:TLAR:NPAX_design"] = inputs["data:TLAR:NPAX"]

        outputs["data:geometry:cabin:seats:passenger:count_by_row"] = inputs[
            "data:geometry:cabin:seats:economical:count_by_row"
        ]

        outputs["data:geometry:cabin:seats:passenger:length"] = inputs[
            "data:geometry:cabin:seats:economical:length"
        ]

        outputs["data:geometry:wing:wet_area"] = inputs["data:geometry:wing:wetted_area"]

        outputs["data:aerodynamics:low_speed:mach"] = inputs[
            "data:aerodynamics:aircraft:takeoff:mach"
        ]

        # outputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"] = (
        # inputs["data:aerodynamics:aircraft:cruise:CL_alpha"]
        # )

        outputs["data:propulsion:he_power_train:fuel_tank:fuel_tank_1:CG:x"] = inputs[
            "data:weight:fuel_tank:CG:x"
        ]

        outputs["data:propulsion:he_power_train:fuel_tank:fuel_tank_2:CG:x"] = inputs[
            "data:weight:fuel_tank:CG:x"
        ]

        outputs["data:mission:sizing:main_route:reserve:altitude"] = inputs[
            "data:mission:sizing:main_route:cruise:altitude"
        ]

        outputs["data:mission:sizing:initial_climb:fuel"] = inputs[
            "data:mission:sizing:main_route:initial_climb:fuel"
        ]

        outputs["data:aerodynamics:wing:cruise:CL0_clean"] = (
            inputs["data:aerodynamics:aircraft:cruise:CL0"] * 1.1
        )  # da correggere

        outputs["data:aerodynamics:wing:low_speed:CL0_clean"] = (
            inputs["data:aerodynamics:aircraft:low_speed:CL0"] * 1.1
        )

        outputs["data:aerodynamics:wing:cruise:CL_alpha"] = inputs[
            "data:aerodynamics:aircraft:cruise:CL_alpha"
        ]

        outputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"] = inputs[
            "data:aerodynamics:aircraft:cruise:induced_drag_coefficient"
        ]

        outputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] = inputs[
            "data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"
        ]

        outputs["data:aerodynamics:wing:low_speed:CL_max_clean"] = inputs[
            "data:aerodynamics:aircraft:low_speed:CL"
        ][-1]

        # propeller

        outputs["data:propulsion:he_power_train:propeller:propeller_1:diameter"] = inputs[
            "data:geometry:propulsion:propeller:diameter"
        ]

        outputs["data:propulsion:he_power_train:propeller:propeller_1:wing_chord_ref"] = inputs[
            "data:geometry:wing:kink:chord"
        ]

        outputs["data:propulsion:he_power_train:propeller:propeller_2:diameter"] = inputs[
            "data:geometry:propulsion:propeller:diameter"
        ]

        outputs["data:propulsion:he_power_train:propeller:propeller_2:wing_chord_ref"] = inputs[
            "data:geometry:wing:kink:chord"
        ]

        # turboshaft

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_1:design_point:T41t"] = (
            inputs["data:propulsion:L1_engine:turbine_inlet_temperature"]
        )

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_1:power_rating"] = inputs[
            "data:propulsion:RTO_power"
        ]

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_1:shaft_power_rating"] = (2160.0486528826723)
        #     inputs["data:propulsion:RTO_power"]
        # )

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_2:design_point:T41t"] = (
            inputs["data:propulsion:L1_engine:turbine_inlet_temperature"]
        )

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_2:power_rating"] = inputs[
            "data:propulsion:RTO_power"
        ]

        outputs["data:propulsion:he_power_train:turboshaft:turboshaft_2:shaft_power_rating"] = (2160.0486528826723)
        #     inputs["data:propulsion:RTO_power"]
        # )

        """outputs['data:aerodynamics:aircraft:cruise:CD0'] = (
            0.027812167420649694
        )

        outputs['data:aerodynamics:aircraft:low_speed:CD0'] = (
            0.02790
        )

        outputs['data:aerodynamics:wing:cruise:CD0'] = (
            0.007942715407273883
        )"""

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:aerodynamics:wing:low_speed:CL_max_clean",
            "data:aerodynamics:aircraft:low_speed:CL",
        ] = np.where(
            inputs["data:aerodynamics:aircraft:low_speed:CL"]
            == np.max(inputs["data:aerodynamics:aircraft:low_speed:CL"]),
            1.0,
            0.0,
        )

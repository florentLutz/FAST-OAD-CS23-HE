# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class CorrectRTANaming(om.ExplicitComponent):
    """
    Naming correction to connect the equivalent variables in both FAST-GA-HE and FAST-OAD-RTA.
    """

    def setup(self):
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:geometry:cabin:seats:economical:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:seats:economical:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:takeoff:mach", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CL_alpha", val=np.nan, units="1/rad")
        self.add_input("data:weight:fuel_tank:CG:x", val=np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
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
        self.add_input("data:geometry:fuselage:wetted_area", val=np.nan, units="m**2")


        self.add_output("data:TLAR:NPAX_design")
        self.add_output("data:geometry:cabin:seats:passenger:count_by_row")
        self.add_output("data:geometry:cabin:seats:passenger:length", units="m")
        self.add_output("data:geometry:wing:wet_area", units="m**2")
        self.add_output("data:aerodynamics:low_speed:mach")
        self.add_output("data:mission:sizing:main_route:reserve:altitude", units="m")
        self.add_output("data:aerodynamics:wing:cruise:CL0_clean")
        self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")
        self.add_output("data:aerodynamics:wing:cruise:CL_alpha")
        self.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient")
        self.add_output("data:aerodynamics:wing:low_speed:induced_drag_coefficient")
        self.add_output("data:aerodynamics:wing:low_speed:CL_max_clean")
        self.add_output("data:geometry:fuselage:wet_area", val=150.0, units="m**2")


    def setup_partials(self):
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
            of="data:mission:sizing:main_route:reserve:altitude",
            wrt="data:mission:sizing:main_route:cruise:altitude",
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
            "data:geometry:fuselage:wet_area", "data:geometry:fuselage:wetted_area", val=1.0
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

        outputs["data:mission:sizing:main_route:reserve:altitude"] = inputs[
            "data:mission:sizing:main_route:cruise:altitude"
        ]

        outputs["data:aerodynamics:wing:cruise:CL0_clean"] = inputs[
            "data:aerodynamics:aircraft:cruise:CL0"
        ]

        outputs["data:aerodynamics:wing:low_speed:CL0_clean"] = inputs[
            "data:aerodynamics:aircraft:low_speed:CL0"
        ]

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

        outputs["data:geometry:fuselage:wet_area"] = inputs["data:geometry:fuselage:wetted_area"]

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

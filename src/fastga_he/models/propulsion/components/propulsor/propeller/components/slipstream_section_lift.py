# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerSectionLift(om.ExplicitComponent):
    """
    Computes the lift coefficient of the section behind the propeller. Will be based on the clean
    wing lift to which we will add the flaps lift increment weighted by the portion of the blown
    span with flaps, when applicable.
    """

    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
            values=["cruise", "takeoff", "landing"],
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]

        self.add_input(name="cl_wing_clean", val=np.nan, shape=number_of_points)
        self.add_input(name="data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)

        self.add_input(name="data:aerodynamics:wing:low_speed:CL_ref", val=np.nan)
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
            val=np.nan,
            desc="Portion of the span, downstream of the propeller, which has flaps",
        )

        shared_inputs = [
            "data:aerodynamics:wing:low_speed:CL_ref",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
        ]

        if flaps_position == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL_2D", val=np.nan)
            shared_inputs += ["data:aerodynamics:flaps:takeoff:CL_2D"]

        elif flaps_position == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL_2D", val=np.nan)
            shared_inputs += ["data:aerodynamics:flaps:landing:CL_2D"]

        self.add_output(
            "unblown_section_lift",
            val=0.5,
            desc="Value of the unblown lift downstream of the propeller",
            shape=number_of_points,
        )
        self.add_output(
            "unblown_section_lift_AOA_0",
            val=0.1,
            desc="Value of the unblown lift downstream of the propeller for a nil AOA",
            shape=number_of_points,
        )

        self.declare_partials(
            of="unblown_section_lift",
            wrt=shared_inputs,
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="unblown_section_lift",
            wrt="cl_wing_clean",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="unblown_section_lift_AOA_0",
            wrt=shared_inputs + ["data:aerodynamics:wing:cruise:CL0_clean"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]
        flaps_position = self.options["flaps_position"]

        cl_wing_clean = inputs["cl_wing_clean"]
        cl_wing_ref = inputs["data:aerodynamics:wing:low_speed:CL_ref"]
        cl_section_ref = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref"
        ]

        cl0 = inputs["data:aerodynamics:wing:cruise:CL0_clean"]

        flapped_ratio = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
        ]

        if flaps_position == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL_2D"]
        elif flaps_position == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL_2D"]
        else:
            delta_cl_flaps = 0.0

        cl_section = cl_section_ref * cl_wing_clean / cl_wing_ref + delta_cl_flaps * flapped_ratio
        cl_section_0 = cl_section_ref * cl0 / cl_wing_ref + delta_cl_flaps * flapped_ratio

        outputs["unblown_section_lift"] = cl_section
        outputs["unblown_section_lift_AOA_0"] = cl_section_0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]

        cl_wing_clean = inputs["cl_wing_clean"]
        cl_wing_ref = inputs["data:aerodynamics:wing:low_speed:CL_ref"]
        cl_section_ref = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref"
        ]

        cl0 = inputs["data:aerodynamics:wing:cruise:CL0_clean"]

        flapped_ratio = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
        ]

        partials["unblown_section_lift", "cl_wing_clean"] = np.full(
            number_of_points, cl_section_ref / cl_wing_ref
        )
        partials["unblown_section_lift_AOA_0", "data:aerodynamics:wing:cruise:CL0_clean"] = np.full(
            number_of_points, cl_section_ref / cl_wing_ref
        )

        partials["unblown_section_lift", "data:aerodynamics:wing:low_speed:CL_ref"] = (
            -cl_wing_clean * cl_section_ref / cl_wing_ref**2.0
        )
        partials["unblown_section_lift_AOA_0", "data:aerodynamics:wing:low_speed:CL_ref"] = (
            -cl0 * cl_section_ref / cl_wing_ref**2.0
        )

        partials[
            "unblown_section_lift",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
        ] = cl_wing_clean / cl_wing_ref
        partials[
            "unblown_section_lift_AOA_0",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":cl_clean_ref",
        ] = cl0 / cl_wing_ref

        if flaps_position == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL_2D"]
            partials["unblown_section_lift", "data:aerodynamics:flaps:takeoff:CL_2D"] = np.full(
                number_of_points, flapped_ratio
            )
            partials["unblown_section_lift_AOA_0", "data:aerodynamics:flaps:takeoff:CL_2D"] = (
                np.full(number_of_points, flapped_ratio)
            )
        elif flaps_position == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL_2D"]
            partials["unblown_section_lift", "data:aerodynamics:flaps:landing:CL_2D"] = np.full(
                number_of_points, flapped_ratio
            )
            partials["unblown_section_lift_AOA_0", "data:aerodynamics:flaps:landing:CL_2D"] = (
                np.full(number_of_points, flapped_ratio)
            )
        else:
            delta_cl_flaps = 0.0

        partials[
            "unblown_section_lift",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
        ] = np.full(number_of_points, delta_cl_flaps)
        partials[
            "unblown_section_lift_AOA_0",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
        ] = np.full(number_of_points, delta_cl_flaps)

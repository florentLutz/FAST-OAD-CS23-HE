# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class AerodynamicDeltasFromPTFile(om.Group):
    """
    Groups that regroups the different computation of aerodynamic deltas and sums them. Also
    contains a subroutine that adds all the deltas that contribute to the wing lift so that the
    lift induced drag increase can be compute afterwards. This means that any lift induced drag
    formula can only be computed here. Also it means we will need a component that computes the
    "clean" aircraft lift regardless of the powertrain.
    """

    def initialize(self):

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

        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]

        self.add_subsystem(
            name="wing_cl_clean",
            subsys=SlipstreamAirframeLiftClean(number_of_points=number_of_points),
            promotes=["*"],
        )

        self.add_subsystem(
            name="wing_cl",
            subsys=SlipstreamAirframeLift(
                number_of_points=number_of_points, flaps_position=flaps_position
            ),
            promotes=["*"],
        )

    # TODO: Promote cl_wing_clean for everyone !


class SlipstreamAirframeLiftClean(om.ExplicitComponent):
    """
    Computation of the wing clean lift. May be required by some components and is also required
    to compute the airframe lift, so we put the computation in common.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(name="alpha", val=np.full(number_of_points, np.nan), units="rad")
        self.add_input(name="data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input(name="data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)

        self.add_output(name="cl_wing_clean", val=0.5, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        alpha = inputs["alpha"]

        cl_wing = cl0_wing + cl_alpha_wing * alpha

        outputs["cl_wing_clean"] = cl_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["cl_wing_clean", "data:aerodynamics:wing:cruise:CL0_clean"] = np.ones(
            number_of_points
        )
        partials["cl_wing_clean", "data:aerodynamics:wing:cruise:CL_alpha"] = inputs["alpha"]
        partials["cl_wing_clean", "alpha"] = (
            np.eye(number_of_points) * inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        )


class SlipstreamAirframeLift(om.ExplicitComponent):
    """
    Computation of the airframe lift as it is required for the computation of the increase in
    lift induced drag. It includes the increase in lift due to the flaps.
    """

    def initialize(self):

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

        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]

        self.add_input(name="cl_wing_clean", val=np.nan, shape=number_of_points)

        if flaps_position == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)

        elif flaps_position == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)

        self.add_output(name="cl_airframe", val=0.5, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        flaps_position = self.options["flaps_position"]

        cl_wing_clean = inputs["cl_wing_clean"]

        if flaps_position == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]

        elif flaps_position == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]

        else:
            delta_cl_flaps = 0.0

        outputs["cl_airframe"] = cl_wing_clean + delta_cl_flaps

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        flaps_position = self.options["flaps_position"]

        partials["cl_airframe", "cl_wing_clean"] = np.eye(number_of_points)

        if flaps_position == "takeoff":
            partials["cl_airframe", "data:aerodynamics:flaps:takeoff:CL"] = np.ones(
                number_of_points
            )

        elif flaps_position == "landing":
            partials["cl_airframe", "data:aerodynamics:flaps:landing:CL"] = np.ones(
                number_of_points
            )

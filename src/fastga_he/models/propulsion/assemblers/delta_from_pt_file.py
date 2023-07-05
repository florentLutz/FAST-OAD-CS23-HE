# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

# noinspection PyUnresolvedReferences
from fastga_he.models.propulsion.components import (
    SlipstreamPropeller,
    SlipstreamPMSM,
    SlipstreamInverter,
    SlipstreamDCBus,
    SlipstreamHarness,
    SlipstreamDCDCConverter,
    SlipstreamBatteryPack,
    SlipstreamDCSSPC,
    SlipstreamDCSplitter,
    SlipstreamRectifier,
    SlipstreamGenerator,
    SlipstreamICE,
)

from .constants import SUBMODEL_THRUST_DISTRIBUTOR


class AerodynamicDeltasFromPTFile(om.Group):
    """
    Groups that regroups the different computation of aerodynamic deltas and sums them. Also
    contains a subroutine that adds all the deltas that contribute to the wing lift so that the
    lift induced drag increase can be compute afterwards. This means that any lift induced drag
    formula can only be computed here. Also it means we will need a component that computes the
    "clean" aircraft lift regardless of the powertrain.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):

        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
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

        self.configurator.load(self.options["power_train_file_path"])

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

        propulsor_names = self.configurator.get_thrust_element_list()
        (
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            components_slipstream_promotes,
            components_slipstream_flap,
        ) = self.configurator.get_slipstream_element_lists()

        options = {
            "power_train_file_path": self.options["power_train_file_path"],
            "number_of_points": number_of_points,
        }
        self.add_subsystem(
            name="thrust_splitter",
            subsys=oad.RegisterSubmodel.get_submodel(SUBMODEL_THRUST_DISTRIBUTOR, options=options),
            promotes=["data:*", "thrust"],
        )

        for (
            component_name,
            component_name_id,
            component_type,
            component_om_type,
            component_slipstream_promotes,
            component_slipstream_flap,
        ) in zip(
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            components_slipstream_promotes,
            components_slipstream_flap,
        ):

            klass = globals()["Slipstream" + component_om_type]
            local_sub_sys = klass()
            local_sub_sys.options[component_name_id] = component_name
            local_sub_sys.options["number_of_points"] = number_of_points
            if component_slipstream_flap:
                local_sub_sys.options["flaps_position"] = flaps_position

            # Because it was more convenient at the time, the "data:*" was chosen to not be
            # universal and thus comes from the SPT field
            self.add_subsystem(
                name=component_name,
                subsys=local_sub_sys,
                promotes_inputs=component_slipstream_promotes,
                promotes_outputs=[],
            )

        for propulsor_name in propulsor_names:
            self.connect(
                "thrust_splitter." + propulsor_name + "_thrust", propulsor_name + ".thrust"
            )


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

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
    SlipstreamFuelTank,
    SlipstreamFuelSystem,
    SlipstreamTurboshaft,
    SlipstreamSpeedReducer,
    SlipstreamPlanetaryGear,
    SlipstreamTurboGenerator,
    SlipstreamGearbox,
    SlipstreamDCAuxLoad,
)

from .constants import (
    SUBMODEL_THRUST_DISTRIBUTOR,
    SUBMODEL_POWER_TRAIN_DELTA_CL,
    SUBMODEL_POWER_TRAIN_DELTA_CM,
    SUBMODEL_POWER_TRAIN_DELTA_CD,
)
from fastga_he.models.performances.mission_vector.constants import HE_SUBMODEL_DEP_EFFECT

DEP_EFFECT_FROM_PT_FILE = "fastga_he.submodel.performances.dep_effect.from_pt_file"


@oad.RegisterSubmodel(HE_SUBMODEL_DEP_EFFECT, DEP_EFFECT_FROM_PT_FILE)
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
            components_slipstream_wing_lift,
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

        self.add_subsystem(
            name="delta_cls_summer",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_POWER_TRAIN_DELTA_CL, options=options
            ),
            promotes=["delta_Cl_wing", "delta_Cl"],
        )
        self.add_subsystem(
            name="delta_cms_summer",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_POWER_TRAIN_DELTA_CM, options=options
            ),
            promotes=["delta_Cm"],
        )
        self.add_subsystem(
            name="delta_cdi",
            subsys=SlipstreamDeltaCdi(number_of_points=number_of_points),
            promotes=["*"],
        )
        self.add_subsystem(
            name="delta_cds_summer",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_POWER_TRAIN_DELTA_CD, options=options
            ),
            promotes_inputs=["delta_Cdi"],
            promotes_outputs=["delta_Cd"],
        )

        for component_name in components_name:
            self.connect(
                component_name + ".delta_Cl",
                "delta_cls_summer." + component_name + "_delta_Cl",
            )
            self.connect(
                component_name + ".delta_Cm",
                "delta_cms_summer." + component_name + "_delta_Cm",
            )
            self.connect(
                component_name + ".delta_Cd",
                "delta_cds_summer." + component_name + "_delta_Cd",
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

        # Need some mock-up interface because the slipstream of some components requires data
        # that other don't. That makes it so that when we need them we need to promote them,
        # when we don't and still promote them it crashes. Hence why the interface
        self.add_input("altitude", val=np.full(number_of_points, np.nan), units="ft")

        self.add_input(name="alpha", val=np.full(number_of_points, np.nan), units="rad")
        self.add_input(name="data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input(name="data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)

        self.add_output(name="cl_wing_clean", val=0.5, shape=number_of_points)

        self.declare_partials(
            of="cl_wing_clean",
            wrt="data:aerodynamics:wing:cruise:CL0_clean",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=np.ones(number_of_points),
        )
        self.declare_partials(
            of="cl_wing_clean",
            wrt="data:aerodynamics:wing:cruise:CL_alpha",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="cl_wing_clean",
            wrt="alpha",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        alpha = inputs["alpha"]

        cl_wing = cl0_wing + cl_alpha_wing * alpha

        outputs["cl_wing_clean"] = cl_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["cl_wing_clean", "data:aerodynamics:wing:cruise:CL_alpha"] = inputs["alpha"]
        partials["cl_wing_clean", "alpha"] = np.full(
            number_of_points, inputs["data:aerodynamics:wing:cruise:CL_alpha"]
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

        self.add_output(name="cl_airframe", val=0.5, shape=number_of_points)

        self.declare_partials(
            of="cl_airframe",
            wrt="cl_wing_clean",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

        if flaps_position == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
            self.declare_partials(
                of="cl_airframe",
                wrt="data:aerodynamics:flaps:takeoff:CL",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
                val=np.ones(number_of_points),
            )

        elif flaps_position == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
            self.declare_partials(
                of="cl_airframe",
                wrt="data:aerodynamics:flaps:landing:CL",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
                val=np.ones(number_of_points),
            )

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


class SlipstreamDeltaCdi(om.ExplicitComponent):
    """
    Computation of the increase in lift induced drag coefficient. Is computed based on the
    delta_Cl on the wing and base on the airframe lift coefficient computed beforehand. Computed
    according to the formula in :cite:`de:2019`
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(name="cl_airframe", val=np.full(number_of_points, np.nan))
        self.add_input(name="delta_Cl_wing", val=np.full(number_of_points, np.nan))
        self.add_input(name="data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        self.add_output(name="delta_Cdi", val=0.0, shape=number_of_points)

        self.declare_partials(
            of="delta_Cdi",
            wrt=["cl_airframe", "delta_Cl_wing"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="delta_Cdi",
            wrt="data:aerodynamics:wing:cruise:induced_drag_coefficient",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl_airframe = inputs["cl_airframe"]
        delta_cl_wing = inputs["delta_Cl_wing"]
        k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        delta_cdi = k * (delta_cl_wing ** 2.0 + 2.0 * cl_airframe * delta_cl_wing)

        outputs["delta_Cdi"] = delta_cdi

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cl_airframe = inputs["cl_airframe"]
        delta_cl_wing = inputs["delta_Cl_wing"]
        k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        partials["delta_Cdi", "cl_airframe"] = 2.0 * k * delta_cl_wing
        partials["delta_Cdi", "delta_Cl_wing"] = 2.0 * k * (delta_cl_wing + cl_airframe)
        partials["delta_Cdi", "data:aerodynamics:wing:cruise:induced_drag_coefficient"] = (
            delta_cl_wing ** 2.0 + 2.0 * cl_airframe * delta_cl_wing
        )
